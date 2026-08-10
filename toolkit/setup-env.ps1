# Toolkit environment: python 3.11, torch CPU, onnxruntime-directml, and a
# pinned checkout of ZFTurbo/Music-Source-Separation-Training (MIT) which
# supplies BOTH the architecture code and the golden reference.
$ErrorActionPreference = 'Stop'

# Pinned so the golden reference and the exported graph never drift apart.
$MsstPin = 'e247dfe4abc1f17c69dff719207fe045dc04413a'   # 2026-07-28

$repo = Split-Path -Parent $PSScriptRoot
Push-Location $repo
try {
    if (-not (Test-Path .venv)) { py -3.11 -m venv .venv }
    .\.venv\Scripts\python.exe -m pip install --upgrade pip
    .\.venv\Scripts\python.exe -m pip install -r toolkit/requirements.txt
    if (-not (Test-Path msst)) {
        git clone https://github.com/ZFTurbo/Music-Source-Separation-Training.git msst
    }
    git -C msst fetch --depth 50 origin $MsstPin
    git -C msst checkout --detach $MsstPin
    New-Item -ItemType Directory -Force checkpoints | Out-Null
    Write-Host "`nready (MSST pinned at $MsstPin). Next: download a checkpoint into checkpoints/ (see README)."
} finally { Pop-Location }
