#!/bin/bash
# ==============================================================================
#  Westwood AI – YOLO Trainer  |  install.sh
#  Run as root:  sudo bash install.sh
# ==============================================================================
set -e

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: This script must be run as root (sudo bash install.sh)."
    exit 1
fi

# ── System packages ────────────────────────────────────────────────────────────
echo ">>> Updating package list..."
apt-get update -y
apt-get upgrade -y
apt-get install -y virtualenv python3-pip python3-dev build-essential unzip wget git

# ── GPU drivers & CUDA (optional) ─────────────────────────────────────────────
read -rp "Install NVIDIA GPU drivers and CUDA toolkit? [y/N] " install_gpu
if [[ "$install_gpu" =~ ^[Yy]$ ]]; then
    echo ">>> Installing GPU drivers and CUDA toolkit..."
    apt-get install -y nvidia-driver-525 nvidia-cuda-toolkit
    echo ">>> GPU drivers instaleeled."
else
    echo ">>> Skipping GPU driver installation."
fi

# ── Virtual environment ────────────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
    echo ">>> Creating virtual environment..."
    virtualenv .venv --python=python3
fi

# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip

# ── Python dependencies ────────────────────────────────────────────────────────
echo ">>> Installing Python dependencies..."
pip install -r requirements.txt

# ── PyTorch with CUDA 11.8 ─────────────────────────────────────────────────────
# Change the --index-url if you need a different CUDA version:
#   CUDA 12.1:  https://download.pytorch.org/whl/cu121
#   CUDA 12.4:  https://download.pytorch.org/whl/cu124
echo ">>> Installing PyTorch (CUDA 11.8)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# ── CUDA sanity check ─────────────────────────────────────────────────────────
echo ">>> Verifying CUDA with PyTorch..."
python - <<'EOF'
import torch
if torch.cuda.is_available():
    print(f"CUDA OK – {torch.cuda.device_count()} device(s): {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: CUDA not available. Check GPU drivers.")
    exit(1)
EOF

# ── Environment file ──────────────────────────────────────────────────────────
if [ ! -f ".env" ]; then
    echo ""
    echo ">>> No .env file found. Copying .env.example → .env ..."
    cp .env.example .env
    echo ">>> IMPORTANT: Edit .env and fill in your ROBOFLOW_API_KEY, AWS_BUCKET_NAME, etc."
else
    echo ">>> .env already exists – skipping copy."
fi

echo ""
echo "============================================================"
echo "  Installation complete!"
echo ""
echo "  Next steps:"
echo "    1. Edit .env with your credentials (never commit this file)"
echo "    2. Activate the environment:  source .venv/bin/activate"
echo "    3. Run training:              python main.py"
echo "    4. See all options:           python main.py --help"
echo "============================================================"
