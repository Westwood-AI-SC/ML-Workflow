"""
main.py – Entry point for the Westwood AI YOLO training pipeline.

Usage examples
--------------
# Minimal – relies entirely on environment variables / .env
python main.py

# Override specific values at the command line
python main.py --epochs 200 --model yolov8s.pt --device 0

# Resume from a checkpoint
python main.py --checkpoint ./weights.pt --epochs 50

# Skip S3 upload (useful for local debugging)
python main.py --no-upload

# Override Roboflow dataset
python main.py --roboflow-project my-project --roboflow-version 3

# Give the run a custom name in S3
python main.py --run-name my-custom-run-v1
"""

import argparse
import logging
import os
import sys
from datetime import datetime

import torch
from roboflow import Roboflow

import config
from src.train_yolo import YOLOTrainer


# ── Logging ───────────────────────────────────────────────────────────────────

def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.FileHandler("main.log"),
            logging.StreamHandler(),
        ],
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Westwood AI – YOLO model training pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Roboflow ──────────────────────────────────────────────
    rf = parser.add_argument_group("Roboflow dataset")
    rf.add_argument(
        "--roboflow-workspace",
        default=config.roboflow_workspace,
        metavar="NAME",
        help="Roboflow workspace slug",
    )
    rf.add_argument(
        "--roboflow-project",
        default=config.roboflow_project,
        metavar="NAME",
        help="Roboflow project slug",
    )
    rf.add_argument(
        "--roboflow-version",
        type=int,
        default=config.roboflow_version,
        metavar="N",
        help="Dataset version number to download",
    )
    rf.add_argument(
        "--roboflow-format",
        default=config.roboflow_export_format,
        metavar="FMT",
        help="Export format (e.g. yolov11, yolov8)",
    )

    # ── Model / training ──────────────────────────────────────
    tr = parser.add_argument_group("Training")
    tr.add_argument(
        "--model",
        default=config.model_path,
        metavar="PATH",
        help="Base YOLO weights to start from (e.g. yolov8n.pt)",
    )
    tr.add_argument(
        "--checkpoint",
        default=config.checkpoint_weights,
        metavar="PATH",
        help="Path to an existing .pt checkpoint to resume training from",
    )
    tr.add_argument(
        "--epochs",
        type=int,
        default=config.epochs,
        metavar="N",
        help="Number of training epochs",
    )
    tr.add_argument(
        "--device",
        type=int,
        default=0,
        metavar="N",
        help="GPU device index (0-based). Use -1 for CPU.",
    )

    # ── AWS S3 ────────────────────────────────────────────────
    s3 = parser.add_argument_group("AWS S3 upload")
    s3.add_argument(
        "--bucket",
        default=config.bucket_name,
        metavar="NAME",
        help="S3 bucket to upload results to",
    )
    s3.add_argument(
        "--region",
        default=config.region_name,
        metavar="REGION",
        help="AWS region of the S3 bucket",
    )
    s3.add_argument(
        "--run-name",
        default=None,
        metavar="NAME",
        help=(
            "Custom name for this run's S3 output folder. "
            "Auto-generated as <project>_v<version>_<timestamp> if omitted."
        ),
    )
    s3.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip uploading results to S3 (useful for local testing)",
    )

    return parser.parse_args()


# ── Pipeline ──────────────────────────────────────────────────────────────────

def main() -> None:
    # Always run relative to this script so paths resolve correctly
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    configure_logging()
    args = parse_args()

    # ── Pre-flight checks ─────────────────────────────────────
    if not config.roboflow_api_key:
        logging.error(
            "ROBOFLOW_API_KEY is not set. "
            "Add it to your .env file or export it as an environment variable."
        )
        sys.exit(1)

    if not args.bucket and not args.no_upload:
        logging.error(
            "AWS_BUCKET_NAME is not set. "
            "Set --bucket, add AWS_BUCKET_NAME to .env, or pass --no-upload to skip S3."
        )
        sys.exit(1)

    if not args.roboflow_workspace or not args.roboflow_project:
        logging.error(
            "Roboflow workspace / project are not set. "
            "Check ROBOFLOW_WORKSPACE and ROBOFLOW_PROJECT in .env or use --roboflow-workspace / --roboflow-project."
        )
        sys.exit(1)

    # ── Build a unique, human-readable run name ───────────────
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name: str = (
        args.run_name
        or f"{args.roboflow_project}_v{args.roboflow_version}_{timestamp}"
    )
    logging.info(f"Run name: {run_name}")

    # ── Step 1: Download dataset from Roboflow ────────────────
    try:
        logging.info("Step 1: Downloading dataset from Roboflow...")
        rf = Roboflow(api_key=config.roboflow_api_key)
        project = rf.workspace(args.roboflow_workspace).project(args.roboflow_project)
        version = project.version(args.roboflow_version)
        dataset = version.download(args.roboflow_format)

        yaml_path = os.path.join(dataset.location, "data.yaml")
        logging.info(f"Dataset downloaded to : {dataset.location}")
        logging.info(f"Using data.yaml       : {yaml_path}")

        # ── Step 2: Initialise trainer ────────────────────────
        trainer = YOLOTrainer(
            model_path=args.model,
            yaml_file=yaml_path,
            epochs=args.epochs,
            bucket_name=args.bucket,
            region_name=args.region,
            checkpoint_weights=args.checkpoint,
            run_name=run_name,
        )

        # ── Step 3: GPU check ─────────────────────────────────
        logging.info("Step 2: Checking CUDA availability...")
        if not torch.cuda.is_available():
            logging.error("CUDA GPU is not available. Exiting.")
            sys.exit(1)
        logging.info(
            f"CUDA available – {torch.cuda.device_count()} device(s) detected. "
            f"Using device {args.device}: {torch.cuda.get_device_name(args.device)}"
        )

        # ── Step 4: Train ─────────────────────────────────────
        logging.info("Step 3: Starting YOLO training...")
        trainer.train_model(device=args.device)

        # ── Step 5: Upload ────────────────────────────────────
        if not args.no_upload:
            zip_path = f"./runs-{run_name}.zip"
            s3_key = f"results/{run_name}.zip"

            logging.info("Step 4: Zipping training results...")
            trainer.zip_results(runs_dir="./runs", zip_path=zip_path)

            logging.info(f"Step 5: Uploading to s3://{args.bucket}/{s3_key} ...")
            trainer.upload_results(zip_path=zip_path, s3_key=s3_key)
            logging.info(f"Results uploaded → s3://{args.bucket}/{s3_key}")
        else:
            logging.info("Skipping S3 upload (--no-upload flag is set).")

        logging.info("=" * 60)
        logging.info("Training pipeline completed successfully.")
        logging.info(f"  Run name : {run_name}")
        if not args.no_upload:
            logging.info(f"  S3 key   : results/{run_name}.zip")
        logging.info("=" * 60)

    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        logging.error(f"An error occurred: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
