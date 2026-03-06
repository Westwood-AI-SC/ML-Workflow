import logging
import time
import zipfile

import boto3


class S3DataHandler:
    """
    Utility class for downloading files from S3 and extracting zip archives.
    """

    def __init__(self, bucket_name: str, region_name: str | None = None) -> None:
        self.bucket_name = bucket_name
        self._s3 = boto3.client("s3", region_name=region_name)

    # ── Download ──────────────────────────────────────────────────────────────

    def download_file(self, s3_key: str, local_path: str) -> None:
        """Download a single file from S3."""
        logging.info(f"Downloading s3://{self.bucket_name}/{s3_key} → {local_path}")
        self._s3.download_file(self.bucket_name, s3_key, local_path)
        logging.info("Download complete.")

    def download_file_with_retry(
        self, s3_key: str, local_path: str, retries: int = 3, delay: int = 5
    ) -> None:
        """Download with exponential-ish back-off retry."""
        for attempt in range(1, retries + 1):
            try:
                self.download_file(s3_key, local_path)
                return
            except Exception as exc:
                if attempt < retries:
                    logging.warning(
                        f"Download attempt {attempt}/{retries} failed: {exc}. "
                        f"Retrying in {delay}s…"
                    )
                    time.sleep(delay)
                else:
                    raise

    # ── Validation ────────────────────────────────────────────────────────────

    def validate_s3_key(self, s3_key: str) -> bool:
        """Return True if the key exists in the bucket, raise otherwise."""
        try:
            self._s3.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except self._s3.exceptions.ClientError as exc:
            if exc.response["Error"]["Code"] == "404":
                raise FileNotFoundError(
                    f"S3 key '{s3_key}' not found in bucket '{self.bucket_name}'."
                )
            raise

    # ── Extraction ────────────────────────────────────────────────────────────

    def extract_zip(self, zip_path: str, extract_to: str = "./") -> None:
        """Extract a zip archive to a local directory."""
        logging.info(f"Extracting '{zip_path}' → '{extract_to}'")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_to)
        logging.info("Extraction complete.")
