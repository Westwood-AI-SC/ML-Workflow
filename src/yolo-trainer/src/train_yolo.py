import logging
import os
import zipfile

import boto3
from ultralytics import YOLO


class YOLOTrainer:
    """
    Handles YOLO model training, result packaging, and S3 upload.

    Parameters
    ----------
    model_path : str
        Base YOLO weights file (e.g. "yolov8n.pt"). Used when no checkpoint is provided.
    yaml_file : str
        Absolute path to the dataset data.yaml file.
    epochs : int
        Number of training epochs.
    bucket_name : str
        S3 bucket to upload results to.
    region_name : str | None
        AWS region (e.g. "us-east-1"). Uses boto3 defaults if None.
    checkpoint_weights : str | None
        Path to an existing .pt checkpoint to resume from. Uses model_path if None.
    run_name : str | None
        Unique identifier for this run, used when naming the S3 object.
    """

    def __init__(
        self,
        model_path: str,
        yaml_file: str,
        epochs: int,
        bucket_name: str,
        region_name: str | None = None,
        checkpoint_weights: str | None = None,
        run_name: str | None = None,
    ) -> None:
        self.model_path = model_path
        self.checkpoint_weights = checkpoint_weights
        self.yaml_file = yaml_file
        self.epochs = epochs
        self.bucket_name = bucket_name
        self.region_name = region_name
        self.run_name = run_name

        # Only create the S3 client when a bucket is configured
        self._s3 = (
            boto3.client("s3", region_name=region_name) if bucket_name else None
        )

    # ── Training ──────────────────────────────────────────────────────────────

    def train_model(self, device: int = 0) -> None:
        """Train the YOLO model."""
        if not os.path.exists(self.yaml_file):
            raise FileNotFoundError(f"data.yaml not found: '{self.yaml_file}'")

        weights_path = self.checkpoint_weights or self.model_path
        if self.checkpoint_weights:
            logging.info(f"Resuming from checkpoint : {weights_path}")
        else:
            logging.info(f"Starting from base model : {weights_path}")

        model = YOLO(weights_path)
        logging.info(
            f"Training for {self.epochs} epochs | data={self.yaml_file} | device={device}"
        )
        model.train(data=self.yaml_file, epochs=self.epochs, device=device)
        logging.info("Training finished.")

    # ── Packaging ─────────────────────────────────────────────────────────────

    def zip_results(self, runs_dir: str = "./runs", zip_path: str = "./runs.zip") -> None:
        """
        Zip the entire YOLO runs directory (all experiments, weights, plots, etc.)
        so the complete training artefact is preserved.
        """
        if not os.path.isdir(runs_dir):
            raise FileNotFoundError(
                f"Runs directory not found: '{runs_dir}'. "
                "Did training complete successfully?"
            )

        logging.info(f"Zipping '{runs_dir}' → '{zip_path}' ...")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for dirpath, _dirnames, files in os.walk(runs_dir):
                for filename in files:
                    abs_path = os.path.join(dirpath, filename)
                    # Store relative to the parent of runs_dir so the zip opens
                    # as  runs/detect/train/weights/best.pt  etc.
                    arc_name = os.path.relpath(abs_path, start=os.path.dirname(runs_dir))
                    zf.write(abs_path, arc_name)

        size_mb = os.path.getsize(zip_path) / (1024 ** 2)
        logging.info(f"Zip created: {zip_path}  ({size_mb:.1f} MB)")

    # ── Upload ────────────────────────────────────────────────────────────────

    def upload_results(self, zip_path: str, s3_key: str) -> None:
        """
        Upload a local zip file to S3.

        The s3_key is expected to be unique per run (e.g.
        ``results/my-project_v16_20260306-143022.zip``) so every training
        run is easy to locate in the bucket.
        """
        if self._s3 is None:
            logging.warning("No S3 client configured – skipping upload.")
            return

        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Zip file not found for upload: '{zip_path}'")

        size_mb = os.path.getsize(zip_path) / (1024 ** 2)
        logging.info(
            f"Uploading {zip_path} ({size_mb:.1f} MB) "
            f"→ s3://{self.bucket_name}/{s3_key}"
        )
        self._s3.upload_file(zip_path, self.bucket_name, s3_key)
        logging.info("Upload complete.")
