# Westwood AI – YOLO Trainer

A clean, public-safe YOLO model training pipeline that pulls datasets from **Roboflow**, trains using **Ultralytics YOLO**, and uploads the full training artefact to **AWS S3** with a unique, timestamped name.

> **No secrets are stored in this repo.** All credentials live in a `.env` file that is gitignored.

---

## Server workflow (SSH → clone → train)

This is the intended end-to-end flow on a fresh EC2 instance:

```bash
# 1. Clone the repo
git clone <repo-url>
cd yolo-trainer

# 2. Install system packages, GPU drivers (optional), venv, and Python deps
sudo ./install.sh
# The script will:
#   - ask whether to install NVIDIA drivers + CUDA
#   - create .venv and install all requirements
#   - copy .env.example → .env automatically

# 3. Activate the virtual environment
source .venv/bin/activate

# 4. Fill in your credentials
nano .env          # or: vi .env

# 5. Run training
python main.py

# See all available CLI flags
python main.py --help
```

### Required `.env` variables

| Variable | Description |
|---|---|
| `ROBOFLOW_API_KEY` | Your Roboflow API key (account settings) |
| `ROBOFLOW_WORKSPACE` | Roboflow workspace slug |
| `ROBOFLOW_PROJECT` | Roboflow project slug |
| `ROBOFLOW_VERSION` | Dataset version number |
| `AWS_BUCKET_NAME` | S3 bucket to upload results to |

Everything else has a sensible default and can be left as-is or overridden via CLI flag.

---

## CLI flags

```
Training:
  --model PATH            Base YOLO weights (default: yolov8n.pt)
  --checkpoint PATH       Resume training from this .pt checkpoint
  --epochs N              Number of training epochs (default: 100)
  --device N              GPU device index, -1 = CPU (default: 0)

Roboflow dataset:
  --roboflow-workspace    Workspace slug
  --roboflow-project      Project slug
  --roboflow-version N    Dataset version number
  --roboflow-format FMT   Export format, e.g. yolov11, yolov8

AWS S3:
  --bucket NAME           S3 bucket name
  --region REGION         AWS region (default: us-east-1)
  --run-name NAME         Custom S3 folder name (auto-generated if omitted)
  --no-upload             Skip uploading to S3 (local testing)
```

### Examples

```bash
# Train for 200 epochs with a larger model
python main.py --epochs 200 --model yolov8s.pt

# Resume from a checkpoint
python main.py --checkpoint ./runs/detect/train/weights/last.pt --epochs 50

# Use a different Roboflow dataset version
python main.py --roboflow-version 17

# Local test – no S3 upload
python main.py --no-upload

# Give the run a custom name in S3
python main.py --run-name production-v2-finetuned
```

---

## S3 output structure

Each run uploads a single zip to:

```
s3://<bucket>/results/<run-name>.zip
```

The run name is auto-generated as `<project>_v<version>_<YYYYMMDD-HHMMSS>` (e.g. `westwoodobjectdetection-bmbpt_v16_20260306-143022`) so every training run is uniquely identifiable in the bucket.

The zip contains the **entire `runs/` directory**, including:
- `runs/detect/train/weights/best.pt`
- `runs/detect/train/weights/last.pt`
- All plots, confusion matrices, and result CSVs

---

## On-server deployment (AWS EC2)

1. SSH into your instance
2. Clone this repo (it is public-safe – no secrets included)
3. Run `sudo bash install.sh`
4. Set environment variables directly on the instance (or copy `.env`)
5. Run `python main.py`
6. Shut down the instance manually when done

### Setting env vars without a .env file (recommended for EC2)

```bash
export ROBOFLOW_API_KEY="..."
export AWS_BUCKET_NAME="..."
# etc.
python main.py
```

Or use AWS Systems Manager Parameter Store / Secrets Manager for production.

---

## Project structure

```
yolo-trainer/
├── main.py              # Entry point + CLI flags
├── config.py            # Reads all config from environment variables
├── requirements.txt
├── install.sh
├── .env.example         # Template – copy to .env and fill in secrets
├── .gitignore           # Ensures .env and artefacts are never committed
└── src/
    ├── train_yolo.py    # YOLOTrainer: train, zip, upload
    ├── data_manager.py  # S3DataHandler: download, extract
    └── sns.py           # SNSNotifier: optional notifications
```
