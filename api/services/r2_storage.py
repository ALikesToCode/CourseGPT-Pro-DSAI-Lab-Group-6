from __future__ import annotations

import logging
from datetime import datetime
from typing import BinaryIO, Dict, List, Optional

import boto3
from botocore.client import Config
from botocore.exceptions import BotoCoreError, ClientError

from api.config import Settings


logger = logging.getLogger(__name__)


class R2StorageService:
    """
    Thin wrapper around Cloudflare R2's S3-compatible API.
    """

    def __init__(self, settings: Settings):
        self._settings = settings
        # Validate required fields up front for clearer errors in dependency injection.
        missing = []
        if not settings.cloudflare_r2_bucket:
            missing.append("CLOUDFLARE_R2_BUCKET_NAME")
        if not settings.cloudflare_r2_endpoint:
            missing.append("CLOUDFLARE_R2_ENDPOINT")
        if not settings.cloudflare_access_key:
            missing.append("CLOUDFLARE_ACCESS_KEY")
        if not settings.cloudflare_secret_access_key:
            missing.append("CLOUDFLARE_SECRET_ACCESS_KEY")
        if missing:
            raise RuntimeError(
                f"Cloudflare R2 is not configured: missing {', '.join(missing)}"
            )

        self.bucket = settings.cloudflare_r2_bucket
        self.endpoint = settings.cloudflare_r2_endpoint.rstrip("/")
        self.region = settings.cloudflare_r2_region or "auto"

        self._client = boto3.client(
            "s3",
            endpoint_url=self.endpoint,
            region_name=self.region,
            aws_access_key_id=settings.cloudflare_access_key,
            aws_secret_access_key=settings.cloudflare_secret_access_key,
            config=Config(signature_version="s3v4"),
        )

    def upload_fileobj(
        self,
        file_obj: BinaryIO,
        key: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """
        Upload an object to the configured R2 bucket.
        """
        extra_args: Dict[str, Dict[str, str]] = {}
        if content_type:
            extra_args["ContentType"] = content_type
        if metadata:
            extra_args["Metadata"] = metadata

        file_obj.seek(0)
        try:
            self._client.upload_fileobj(file_obj, self.bucket, key, ExtraArgs=extra_args or None)
            logger.info("Uploaded %s to bucket %s", key, self.bucket)
        except (ClientError, BotoCoreError) as exc:
            logger.exception("Failed to upload %s to R2", key)
            raise RuntimeError("Failed to upload file to Cloudflare R2") from exc

        return {"key": key, "url": self.build_object_url(key)}

    def list_objects(self, prefix: Optional[str] = None, max_items: int = 500) -> List[Dict[str, Optional[str]]]:
        """
        List objects in the bucket. Used by the /files endpoint to display the latest uploads.
        """
        params = {"Bucket": self.bucket, "MaxKeys": max_items}
        if prefix:
            params["Prefix"] = prefix

        try:
            response = self._client.list_objects_v2(**params)
        except (ClientError, BotoCoreError) as exc:
            logger.exception("Failed to list objects from R2 (region=%s, endpoint=%s)", self.region, self.endpoint)
            raise RuntimeError("Unable to list files from Cloudflare R2") from exc

        contents = response.get("Contents", []) or []
        return [
            {
                "key": obj.get("Key"),
                "size": obj.get("Size"),
                "etag": obj.get("ETag"),
                "last_modified": obj.get("LastModified"),
                "url": self.build_object_url(obj.get("Key", "")),
            }
            for obj in contents
        ]

    def delete_object(self, key: str) -> None:
        """
        Delete an object from the R2 bucket.
        """
        try:
            self._client.delete_object(Bucket=self.bucket, Key=key)
            logger.info("Deleted %s from bucket %s", key, self.bucket)
        except (ClientError, BotoCoreError) as exc:
            logger.exception("Failed to delete %s from R2", key)
            raise RuntimeError("Failed to delete file from Cloudflare R2") from exc

    def generate_presigned_get_url(self, key: str, expires_in: int = 900) -> str:
        """
        Generate a temporary, signed URL that can be used to view/download the
        specified object.
        """
        try:
            return self._client.generate_presigned_url(
                ClientMethod="get_object",
                Params={"Bucket": self.bucket, "Key": key},
                ExpiresIn=expires_in,
            )
        except (ClientError, BotoCoreError) as exc:
            logger.exception("Failed to generate presigned URL for %s", key)
            raise RuntimeError("Unable to generate presigned URL") from exc

    def build_object_url(self, key: str) -> str:
        """
        Construct the public-style URL for an object. Buckets can be made public
        via Cloudflare dashboard or by generating signed URLs elsewhere.
        """
        key = key.lstrip("/")
        return f"{self.endpoint}/{self.bucket}/{key}"
