import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv


load_dotenv()


def _env(*keys: str, default: Optional[str] = None) -> Optional[str]:
    """
    Return the first environment variable value that exists for the provided keys.
    Accepts keys that use either '-' or '_' as separators so we can support the
    screenshot-style .env names without forcing users to rename them.
    """
    normalized = list(keys)
    for key in list(keys):
        normalized.append(key.replace("-", "_"))
        normalized.append(key.replace("_", "-"))

    for key in normalized:
        value = os.getenv(key)
        if value:
            return value
    return default


class Settings:
    """
    Central application configuration holder so we don't need to keep calling
    os.getenv throughout the codebase.
    """

    def __init__(self) -> None:
        self.cloudflare_ai_search_token = _env("CLOUDFLARE_AI_SEARCH_TOKEN", "CLOUDFLARE-AI-SEARCH-TOKEN")
        self.cloudflare_account_id = _env("CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE-ACCOUNT-ID")
        self.cloudflare_rag_id = _env("CLOUDFLARE_RAG_ID", "CLOUDFLARE-AI-SEARCH-RAG-ID")

        self.cloudflare_access_key = _env("CLOUDFLARE_ACCESS_KEY", "CLOUDFLARE-ACCESS-KEY", "CLOUDFLARE-ACESS-KEY")
        self.cloudflare_secret_access_key = _env(
            "CLOUDFLARE_SECRET_ACCESS_KEY",
            "CLOUDFLARE-SECRET-ACCESS-KEY",
            "CLOUDFLARE-SECRET-ACESS-KEY",
        )
        self.cloudflare_r2_bucket = _env("CLOUDFLARE_R2_BUCKET_NAME", "CLOUDFLARE-R2-BUCKET-NAME")
        self.cloudflare_r2_endpoint = _env("CLOUDFLARE_R2_ENDPOINT", "CLOUDFLARE-R2-ENDPOINT")
        self.cloudflare_r2_token = _env("CLOUDFLARE_R2_TOKEN", "CLOUDFLARE-R2-TOKEN")

        # Basic validation for required settings so failures happen on startup rather than at runtime.
        required = {
            "CLOUDFLARE_ACCESS_KEY": self.cloudflare_access_key,
            "CLOUDFLARE_SECRET_ACCESS_KEY": self.cloudflare_secret_access_key,
            "CLOUDFLARE_R2_BUCKET_NAME": self.cloudflare_r2_bucket,
            "CLOUDFLARE_R2_ENDPOINT": self.cloudflare_r2_endpoint,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise RuntimeError(
                f"Missing required Cloudflare R2 settings: {', '.join(missing)}. "
                "Double-check your .env file."
            )

    @property
    def has_ai_search(self) -> bool:
        """
        Returns True when the configuration contains enough information to
        talk to Cloudflare AI Search (AutoRAG) APIs.
        """
        return bool(self.cloudflare_ai_search_token and self.cloudflare_account_id and self.cloudflare_rag_id)


@lru_cache
def get_settings() -> Settings:
    """
    Cached accessor so Settings is instantiated once per process.
    """
    return Settings()

