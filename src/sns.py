import logging

import boto3


class SNSNotifier:
    """
    Optional SNS notification helper.

    Usage
    -----
    Pass ``sns_topic_arn`` from config (or None to disable).
    The notifier is a no-op when topic_arn is None so callers
    don't need to guard every call site.
    """

    def __init__(self, topic_arn: str | None, region_name: str | None = None) -> None:
        self.topic_arn = topic_arn
        self._client = (
            boto3.client("sns", region_name=region_name) if topic_arn else None
        )

    def notify(self, message: str, subject: str = "Westwood AI – Training Update") -> None:
        """Send an SNS message. Silently skipped when topic_arn is not set."""
        if self._client is None or not self.topic_arn:
            logging.debug("SNS not configured – notification skipped.")
            return

        logging.info(f"Sending SNS notification to {self.topic_arn}…")
        response = self._client.publish(
            TopicArn=self.topic_arn,
            Subject=subject,
            Message=message,
        )
        logging.info(f"SNS message sent. MessageId: {response['MessageId']}")
