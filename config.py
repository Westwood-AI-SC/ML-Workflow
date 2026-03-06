"""
config.py – All configuration is read from environment variables.

For local development:
    1. Copy .env.example to .env
    2. Fill in your values in .env
    3. Run the trainer – python-dotenv will pick up .env automatically.

On AWS (EC2 / ECS / etc.):
    Set environment variables directly on the instance / task definition.
    Do NOT copy .env to the server.
"""

import os
from dotenv import load_dotenv

# Load .env when running locally. No-op if the file doesn't exist.
load_dotenv()

# ── Roboflow ──────────────────────────────────────────────────────────────────
roboflow_api_key: str       = os.environ.get("ROBOFLOW_API_KEY", "")
roboflow_workspace: str     = os.environ.get("ROBOFLOW_WORKSPACE", "")
roboflow_project: str       = os.environ.get("ROBOFLOW_PROJECT", "")
roboflow_version: int       = int(os.environ.get("ROBOFLOW_VERSION", "1"))
roboflow_export_format: str = os.environ.get("ROBOFLOW_EXPORT_FORMAT", "yolov11")

# ── AWS S3 ────────────────────────────────────────────────────────────────────
bucket_name: str  = os.environ.get("AWS_BUCKET_NAME", "")
region_name: str  = os.environ.get("AWS_REGION", "us-east-1")

# ── SNS (optional) ────────────────────────────────────────────────────────────
# Leave SNS_TOPIC_ARN blank / unset to disable notifications entirely.
sns_topic_arn: str | None = os.environ.get("SNS_TOPIC_ARN") or None

# ── Training ──────────────────────────────────────────────────────────────────
model_path: str            = os.environ.get("MODEL_PATH", "yolov8n.pt")
epochs: int                = int(os.environ.get("EPOCHS", "100"))
# Path to an existing .pt checkpoint to resume from; None = start fresh.
checkpoint_weights: str | None = os.environ.get("CHECKPOINT_WEIGHTS") or None
