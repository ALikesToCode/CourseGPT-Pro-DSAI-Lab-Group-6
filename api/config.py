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
        # Cloudflare uses non-AWS region names (e.g., auto, wnam, apac). Default to "auto" to avoid
        # inheriting AWS_REGION/DEFAULT_REGION values like "ap-south-1" that will be rejected by R2.
        self.cloudflare_r2_region = _env("CLOUDFLARE_R2_REGION", default="auto")
        self.cloudflare_r2_token = _env("CLOUDFLARE_R2_TOKEN", "CLOUDFLARE-R2-TOKEN")

        # Agent Configuration
        self.router_agent_url = _env("ROUTER_AGENT_URL")
        self.router_agent_api_key = _env("ROUTER_AGENT_API_KEY")
        self.router_agent_model = _env("ROUTER_AGENT_MODEL")

        self.code_agent_url = _env("CODE_AGENT_URL")
        self.code_agent_api_key = _env("CODE_AGENT_API_KEY")
        self.code_agent_model = _env("CODE_AGENT_MODEL")

        self.math_agent_url = _env("MATH_AGENT_URL")
        self.math_agent_api_key = _env("MATH_AGENT_API_KEY")
        self.math_agent_model = _env("MATH_AGENT_MODEL")

        self.general_agent_url = _env("GENERAL_AGENT_URL")
        self.general_agent_api_key = _env("GENERAL_AGENT_API_KEY")
        self.general_agent_model = _env("GENERAL_AGENT_MODEL")

        self.google_api_key = _env("GOOGLE_API_KEY", "GEMINI_API_KEY")
        self.gemini_model = _env("GEMINI_MODEL", default="gemini-3-pro-preview")
        self.vertex_project_id = _env("VERTEX_PROJECT_ID", "GCP_PROJECT", "GOOGLE_CLOUD_PROJECT")
        self.vertex_location = _env("VERTEX_LOCATION", default="us-central1")
        self.vertex_credentials_b64 = _env("VERTEX_CREDENTIALS_JSON_B64", "VERTEX_CREDENTIALS_BASE64")
        
        self.gemini_fallback_models = []
        i = 1
        while True:
            model = _env(f"GEMINI_MODEL_{i}")
            if not model:
                break
            self.gemini_fallback_models.append(model)
            i += 1

        # Basic validation for required settings so failures happen on startup rather than at runtime.
        # We make R2 optional to allow the API to start even if R2 is not configured.
        required = {
            # "CLOUDFLARE_ACCESS_KEY": self.cloudflare_access_key,
            # "CLOUDFLARE_SECRET_ACCESS_KEY": self.cloudflare_secret_access_key,
            # "CLOUDFLARE_R2_BUCKET_NAME": self.cloudflare_r2_bucket,
            # "CLOUDFLARE_R2_ENDPOINT": self.cloudflare_r2_endpoint,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            print(f"Warning: Missing R2 settings: {', '.join(missing)}. File uploads will not work.")
            # raise RuntimeError(
            #     f"Missing required Cloudflare R2 settings: {', '.join(missing)}. "
            #     "Double-check your .env file."
            # )

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
