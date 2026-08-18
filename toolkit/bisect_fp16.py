"""¿Existe ALGÚN conjunto de nodos en fp32 que recupere la precisión de kim?

Bloquear por tipo de operación ya se probó y no alcanza: RMSNorm evita el NaN
pero deja 1.25 dB de pérdida, y agregar atención, MatMul o el GLU no mueve la
aguja (ver `docs/fp16-findings.md`). Eso sugiere que la pérdida está repartida,
pero "sugiere" no es "está medido": puede haber un puñado de nodos concretos que
la causen y que ningún tipo de operación agrupe bien.

Esto lo responde por bisección sobre los nodos, no por intuición. El buscador de
la librería no sirve —aborta en su propio chequeo previo, con NADA convertido—,
así que la búsqueda se hace acá, y es simple a propósito:

  1. Se mide el error del candidato sin bloquear nada (más allá del mínimo).
  2. Si no alcanza, se parte la lista de nodos al medio y se prueba bloqueando
     cada mitad. La mitad que más baja el error se queda, y se recursa adentro.
  3. Termina cuando el error entra en el objetivo, o cuando bloquear la mitad ya
     no mejora — que también es una respuesta: la pérdida está repartida.

El error se mide contra la salida del grafo fp32 (no contra el golden de torch):
es la comparación directa de "lo mismo pero en otra precisión", y es barata —
una sola pasada del chunk 0.

    python toolkit/bisect_fp16.py mel_band_roformer_kim
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO / "artifacts"
GOLDEN = REPO / "refs" / "golden"
TRABAJO = ARTIFACTS / "bisect"

# Sin estos cuatro la salida es NaN entera (RMSNorm desborda el techo de fp16), y
# sin ScatterElements el grafo ni corre en mel_band: no hay kernel fp16 de CPU
# para reduction='add'. Son el piso, no parte de la búsqueda.
PISO = ["Pow", "ReduceMean", "Sqrt", "Div", "ScatterElements"]

# Una décima parte del error que deja el piso solo (p99.9 3.8e-2). Si ni
# bloqueando la mitad del modelo se llega acá, la pérdida no está localizada.
OBJETIVO_P999 = 3.0e-3


def source_graph(name: str) -> Path:
    manifest = json.loads((ARTIFACTS / "manifest.json").read_text())
    return ARTIFACTS / manifest["models"][name]["file"]


def nodos_convertibles(src: Path) -> list[str]:
    """Los nodos que la conversión tocaría, en orden topológico.

    Se excluyen los del piso: bloquearlos ya es la base de toda la búsqueda, y
    dejarlos entrar haría que la bisección los "descubra" una y otra vez.
    """
    import onnx

    modelo = onnx.load(str(src), load_external_data=False)
    return [n.name for n in modelo.graph.node if n.op_type not in PISO and n.name]


def convertir(src: Path, destino: Path, bloqueados: list[str]):
    import onnx
    from onnxruntime.transformers import float16
    from onnxruntime.transformers.onnx_model import OnnxModel

    destino.unlink(missing_ok=True)
    (destino.parent / (destino.name + ".data")).unlink(missing_ok=True)
    convertido = float16.convert_float_to_float16(
        onnx.load(str(src)),
        keep_io_types=True,
        op_block_list=PISO,
        node_block_list=bloqueados or None,
    )
    del convertido.graph.value_info[:]
    envoltorio = OnnxModel(convertido)
    envoltorio.topological_sort()
    onnx.save(
        envoltorio.model,
        str(destino),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=destino.name + ".data",
    )
    return destino


def error_de(destino: Path, spec: np.ndarray, referencia: np.ndarray) -> dict:
    import onnxruntime as ort

    ort.set_default_logger_severity(3)
    sesion = ort.InferenceSession(str(destino), providers=["DmlExecutionProvider"])
    salida = sesion.run(None, {"spec": spec})[0]
    diferencia = np.abs(referencia.astype(np.float64) - salida.astype(np.float64))
    return {
        "max": float(diferencia.max()),
        "rms": float(np.sqrt((diferencia**2).mean())),
        "p999": float(np.percentile(diferencia, 99.9)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--target", type=float, default=OBJETIVO_P999)
    ap.add_argument("--max-iters", type=int, default=24)
    args = ap.parse_args()

    import onnxruntime as ort

    src = source_graph(args.name)
    TRABAJO.mkdir(parents=True, exist_ok=True)
    candidato = TRABAJO / f"{src.stem}_bisect.onnx"

    spec = np.load(GOLDEN / args.name / "chunk0_spec.npy")
    ort.set_default_logger_severity(3)
    referencia = ort.InferenceSession(
        str(src), providers=["DmlExecutionProvider"]
    ).run(None, {"spec": spec})[0]
    print(f"referencia fp32 lista, salida {referencia.shape}")

    nodos = nodos_convertibles(src)
    print(f"{len(nodos)} nodos candidatos a quedarse en fp32\n")

    def medir(bloqueados: list[str], etiqueta: str) -> dict:
        arranque = time.perf_counter()
        convertir(src, candidato, bloqueados)
        e = error_de(candidato, spec, referencia)
        tam = candidato.stat().st_size + (candidato.parent / (candidato.name + ".data")).stat().st_size
        print(f"  {etiqueta:34s} p99.9={e['p999']:.3e} max={e['max']:.3e} "
              f"{tam/2**20:6.1f} MiB  ({time.perf_counter()-arranque:.0f}s)")
        return e

    base = medir([], "sin bloquear nada extra")
    if base["p999"] <= args.target:
        print("\nEl piso ya alcanza: no hace falta bloquear nada mas.")
        return

    bloqueados: list[str] = []
    ventana = nodos
    for i in range(args.max_iters):
        if len(ventana) <= 1:
            print("\nLa ventana se redujo a un nodo: no hay mas que partir.")
            break
        mitad = len(ventana) // 2
        izquierda, derecha = ventana[:mitad], ventana[mitad:]
        e_izq = medir(bloqueados + izquierda, f"[{i}] bloqueo primera mitad ({len(izquierda)})")
        e_der = medir(bloqueados + derecha, f"[{i}] bloqueo segunda mitad ({len(derecha)})")

        mejor, ventana_nueva = (
            (e_izq, izquierda) if e_izq["p999"] < e_der["p999"] else (e_der, derecha)
        )
        if mejor["p999"] >= base["p999"] * 0.9:
            print("\nBloquear media red no baja el error ni un 10%: "
                  "la perdida esta REPARTIDA, no en un puñado de nodos.")
            return
        if mejor["p999"] <= args.target:
            print(f"\nObjetivo alcanzado bloqueando {len(ventana_nueva)} nodos; "
                  "se sigue achicando para encontrar el minimo.")
        base = mejor
        ventana = ventana_nueva

    print(f"\nMejor error alcanzado: p99.9={base['p999']:.3e} "
          f"con {len(ventana)} nodos en fp32 (objetivo {args.target:.1e}).")


if __name__ == "__main__":
    main()
