#!/bin/bash
# ==============================================================================
#  Westwood AI – YOLO Trainer  |  install.sh
#  Run once on a fresh Ubuntu server:  sudo bash install.sh
# ==============================================================================
set -e

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: Run as root → sudo bash install.sh"
    exit 1
fi

# ── System packages ────────────────────────────────────────────────────────────
echo ">>> Updating packages..."
apt-get update -y
apt-get upgrade -y
apt-get install -y virtualenv python3-pip python3-dev build-essential \
    unzip wget git ubuntu-drivers-common

# ── GPU drivers (Ubuntu auto-detect) ──────────────────────────────────────────
read -rp "Install GPU drivers? (recommended for EC2 GPU instances) [y/N] " install_gpu
if [[ "$install_gpu" =~ ^[Yy]$ ]]; then
    echo ">>> Running ubuntu-drivers autoinstall..."
    ubuntu-drivers autoinstall
    echo ">>> GPU drivers installed. A reboot may be required."
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

# ── PyTorch with CUDA ─────────────────────────────────────────────────────────
# ubuntu-drivers picks the best driver; we match PyTorch to CUDA 11.8 by default.
# Change the --index-url if your driver supports a newer CUDA version:
#   CUDA 12.1 → https://download.pytorch.org/whl/cu121
#   CUDA 12.4 → https://download.pytorch.org/whl/cu124
echo ">>> Installing PyTorch (CUDA 11.8)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# ── CUDA sanity check ─────────────────────────────────────────────────────────
echo ">>> Verifying CUDA..."
python - <<'EOF'
import torch
if torch.cuda.is_available():
    print(f"CUDA OK – {torch.cuda.device_count()} device(s): {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: CUDA not available. Check GPU drivers (reboot may be needed).")
EOF

# ── Environment file ──────────────────────────────────────────────────────────
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo ">>> .env created from .env.example"
else
    echo ">>> .env already exists – skipping copy."
fi

echo ""
echo "============================================================"
echo "  Done! Next steps:"
echo ""
echo "  1. Activate the venv:   source .venv/bin/activate"
echo "  2. Edit credentials:    nano .env"
echo "  3. Run training:        python main.py"
echo "  4. All CLI options:     python main.py --help"
echo "============================================================"

