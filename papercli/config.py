"""Configuration management for papercli."""

import os
from pathlib import Path
from typing import Optional

import toml
from pydantic import BaseModel, Field, PrivateAttr


class LLMConfig(BaseModel):
    """LLM configuration."""

    base_url: str = Field(default="https://api.openai.com/v1")
    api_key: Optional[str] = Field(default=None)
    reasoning_model: str = Field(default="gpt-4o-mini")
    instinct_model: str = Field(default="gpt-4o")
    timeout: float = Field(default=60.0)
    max_retries: int = Field(default=3)


class CacheConfig(BaseModel):
    """Cache configuration."""

    path: str = Field(default="~/.cache/papercli.sqlite")
    enabled: bool = Field(default=True)
    ttl_hours: int = Field(default=24 * 7)  # 1 week default


class Doc2XConfig(BaseModel):
    """Doc2X PDF parsing service configuration."""

    base_url: str = Field(default="https://v2.doc2x.noedgeai.com")
    api_key: Optional[str] = Field(default=None)
    timeout: float = Field(default=60.0)  # HTTP request timeout
    poll_interval: float = Field(default=2.0)  # Polling interval in seconds
    max_wait: float = Field(default=900.0)  # Max wait time (15 min default)


class UnpaywallConfig(BaseModel):
    """Unpaywall API configuration for PDF fetching."""

    base_url: str = Field(default="https://api.unpaywall.org/v2")
    email: Optional[str] = Field(default=None)  # Required for polite pool
    timeout: float = Field(default=30.0)


class APIKeysConfig(BaseModel):
    """API keys configuration."""

    serpapi_key: Optional[str] = Field(default=None)
    ncbi_api_key: Optional[str] = Field(default=None)  # Optional for PubMed
    openalex_email: Optional[str] = Field(default=None)  # Polite pool for OpenAlex


class GeminiConfig(BaseModel):
    """Gemini API configuration for slide generation."""

    base_url: str = Field(default="https://api.openai-proxy.org/google/v1beta")
    api_key: Optional[str] = Field(default=None)
    text_model: str = Field(default="gemini-3-flash-preview")
    image_model: str = Field(default="gemini-3-pro-image-preview")
    timeout: float = Field(default=120.0)  # Longer timeout for image generation
    max_retries: int = Field(default=3)


class ZoteroConfig(BaseModel):
    """Zotero API configuration for personal library search."""

    base_url: str = Field(default="https://api.zotero.org")
    api_key: Optional[str] = Field(default=None)
    user_id: Optional[str] = Field(default=None)
    qmode: str = Field(default="titleCreatorYear")  # Query mode: titleCreatorYear or everything
    item_type: str = Field(default="-attachment")  # Item type filter (- prefix excludes)
    timeout: float = Field(default=30.0)


class Settings(BaseModel):
    """Main settings container."""

    _config_file_used: Optional[Path] = PrivateAttr(default=None)

    llm: LLMConfig = Field(default_factory=LLMConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    api_keys: APIKeysConfig = Field(default_factory=APIKeysConfig)
    doc2x: Doc2XConfig = Field(default_factory=Doc2XConfig)
    unpaywall: UnpaywallConfig = Field(default_factory=UnpaywallConfig)
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)
    zotero: ZoteroConfig = Field(default_factory=ZoteroConfig)

    # CLI overrides (set directly from command line)
    reasoning_model: Optional[str] = Field(default=None)
    instinct_model: Optional[str] = Field(default=None)
    llm_base_url: Optional[str] = Field(default=None)
    cache_path: Optional[str] = Field(default=None)
    cache_enabled: bool = Field(default=True)

    def __init__(self, **data):
        super().__init__(**data)
        self._load_from_env()
        self._load_from_config_file()
        self._apply_cli_overrides()

    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        # LLM API key (LLM_API_KEY only)
        if api_key := os.environ.get("LLM_API_KEY"):
            self.llm.api_key = api_key

        # LLM base URL (LLM_BASE_URL only)
        if base_url := os.environ.get("LLM_BASE_URL"):
            self.llm.base_url = base_url

        # Model overrides from env
        if reasoning_model := os.environ.get("PAPERCLI_REASONING_MODEL"):
            self.llm.reasoning_model = reasoning_model
        if instinct_model := os.environ.get("PAPERCLI_INSTINCT_MODEL"):
            self.llm.instinct_model = instinct_model

        # API keys
        if serpapi_key := os.environ.get("SERPAPI_API_KEY"):
            self.api_keys.serpapi_key = serpapi_key
        if ncbi_key := os.environ.get("NCBI_API_KEY"):
            self.api_keys.ncbi_api_key = ncbi_key
        if openalex_email := os.environ.get("OPENALEX_EMAIL"):
            self.api_keys.openalex_email = openalex_email

        # Cache path
        if cache_path := os.environ.get("PAPERCLI_CACHE_PATH"):
            self.cache.path = cache_path

        # Doc2X configuration
        if doc2x_api_key := os.environ.get("DOC2X_API_KEY"):
            self.doc2x.api_key = doc2x_api_key
        if doc2x_base_url := os.environ.get("DOC2X_BASE_URL"):
            self.doc2x.base_url = doc2x_base_url

        # Unpaywall configuration
        if unpaywall_email := os.environ.get("UNPAYWALL_EMAIL"):
            self.unpaywall.email = unpaywall_email

        # Gemini configuration
        if gemini_api_key := os.environ.get("GEMINI_API_KEY"):
            self.gemini.api_key = gemini_api_key
        if gemini_base_url := os.environ.get("GEMINI_BASE_URL"):
            self.gemini.base_url = gemini_base_url
        if gemini_text_model := os.environ.get("GEMINI_TEXT_MODEL"):
            self.gemini.text_model = gemini_text_model
        if gemini_image_model := os.environ.get("GEMINI_IMAGE_MODEL"):
            self.gemini.image_model = gemini_image_model

        # Zotero configuration
        if zotero_api_key := os.environ.get("ZOTERO_API_KEY"):
            self.zotero.api_key = zotero_api_key
        if zotero_user_id := os.environ.get("ZOTERO_USER_ID"):
            self.zotero.user_id = zotero_user_id
        if zotero_base_url := os.environ.get("ZOTERO_BASE_URL"):
            self.zotero.base_url = zotero_base_url

    def _load_from_config_file(self) -> None:
        """Load configuration from config file (~/.papercli.toml)."""
        config_paths = [
            Path.home() / ".papercli.toml",
            Path.home() / ".config" / "papercli" / "config.toml",
            Path("papercli.toml"),  # Current directory
        ]

        for config_path in config_paths:
            if config_path.exists():
                try:
                    config_data = toml.load(config_path)
                    self._merge_config(config_data)
                    self._config_file_used = config_path
                    break
                except Exception:
                    pass  # Ignore invalid config files

    def get_config_file_used(self) -> Optional[Path]:
        """Return which config file path was loaded (if any)."""
        return self._config_file_used

    def _merge_config(self, config_data: dict) -> None:
        """Merge config file data into settings."""
        if llm_config := config_data.get("llm"):
            if "api_key" in llm_config and not self.llm.api_key:
                self.llm.api_key = llm_config["api_key"]
            if "base_url" in llm_config and not self.llm_base_url:
                self.llm.base_url = llm_config["base_url"]
            if "reasoning_model" in llm_config and not self.reasoning_model:
                self.llm.reasoning_model = llm_config["reasoning_model"]
            if "instinct_model" in llm_config and not self.instinct_model:
                self.llm.instinct_model = llm_config["instinct_model"]
            if "timeout" in llm_config:
                self.llm.timeout = llm_config["timeout"]
            if "max_retries" in llm_config:
                self.llm.max_retries = llm_config["max_retries"]

        if cache_config := config_data.get("cache"):
            if "path" in cache_config and not self.cache_path:
                self.cache.path = cache_config["path"]
            if "enabled" in cache_config:
                self.cache.enabled = cache_config["enabled"]
            if "ttl_hours" in cache_config:
                self.cache.ttl_hours = cache_config["ttl_hours"]

        if api_keys := config_data.get("api_keys"):
            if "serpapi_key" in api_keys and not self.api_keys.serpapi_key:
                self.api_keys.serpapi_key = api_keys["serpapi_key"]
            if "ncbi_api_key" in api_keys and not self.api_keys.ncbi_api_key:
                self.api_keys.ncbi_api_key = api_keys["ncbi_api_key"]
            if "openalex_email" in api_keys and not self.api_keys.openalex_email:
                self.api_keys.openalex_email = api_keys["openalex_email"]

        if doc2x_config := config_data.get("doc2x"):
            if "api_key" in doc2x_config and not self.doc2x.api_key:
                self.doc2x.api_key = doc2x_config["api_key"]
            if "base_url" in doc2x_config:
                self.doc2x.base_url = doc2x_config["base_url"]
            if "timeout" in doc2x_config:
                self.doc2x.timeout = doc2x_config["timeout"]
            if "poll_interval" in doc2x_config:
                self.doc2x.poll_interval = doc2x_config["poll_interval"]
            if "max_wait" in doc2x_config:
                self.doc2x.max_wait = doc2x_config["max_wait"]

        if unpaywall_config := config_data.get("unpaywall"):
            if "email" in unpaywall_config and not self.unpaywall.email:
                self.unpaywall.email = unpaywall_config["email"]
            if "base_url" in unpaywall_config:
                self.unpaywall.base_url = unpaywall_config["base_url"]
            if "timeout" in unpaywall_config:
                self.unpaywall.timeout = unpaywall_config["timeout"]

        if gemini_config := config_data.get("gemini"):
            if "api_key" in gemini_config and not self.gemini.api_key:
                self.gemini.api_key = gemini_config["api_key"]
            if "base_url" in gemini_config:
                self.gemini.base_url = gemini_config["base_url"]
            if "text_model" in gemini_config:
                self.gemini.text_model = gemini_config["text_model"]
            if "image_model" in gemini_config:
                self.gemini.image_model = gemini_config["image_model"]
            if "timeout" in gemini_config:
                self.gemini.timeout = gemini_config["timeout"]
            if "max_retries" in gemini_config:
                self.gemini.max_retries = gemini_config["max_retries"]

        if zotero_config := config_data.get("zotero"):
            if "api_key" in zotero_config and not self.zotero.api_key:
                self.zotero.api_key = zotero_config["api_key"]
            if "user_id" in zotero_config and not self.zotero.user_id:
                self.zotero.user_id = zotero_config["user_id"]
            if "base_url" in zotero_config:
                self.zotero.base_url = zotero_config["base_url"]
            if "qmode" in zotero_config:
                self.zotero.qmode = zotero_config["qmode"]
            if "item_type" in zotero_config:
                self.zotero.item_type = zotero_config["item_type"]
            if "timeout" in zotero_config:
                self.zotero.timeout = zotero_config["timeout"]

    def _apply_cli_overrides(self) -> None:
        """Apply CLI-provided overrides."""
        if self.reasoning_model:
            self.llm.reasoning_model = self.reasoning_model
        if self.instinct_model:
            self.llm.instinct_model = self.instinct_model
        if self.llm_base_url:
            self.llm.base_url = self.llm_base_url
        if self.cache_path:
            self.cache.path = self.cache_path
        if not self.cache_enabled:
            self.cache.enabled = False

    def get_reasoning_model(self) -> str:
        """Get the model to use for reasoning tasks (intent extraction, query rewriting, segmentation)."""
        return self.reasoning_model or self.llm.reasoning_model

    def get_instinct_model(self) -> str:
        """Get the model to use for instinct tasks (evaluation, reranking, review critique)."""
        return self.instinct_model or self.llm.instinct_model

    def get_cache_path(self) -> Path:
        """Get the resolved cache path."""
        path = self.cache_path or self.cache.path
        return Path(path).expanduser()

    def get_llm_api_key(self) -> str:
        """Get LLM API key or raise error."""
        if not self.llm.api_key:
            raise ValueError(
                "LLM API key not configured. Set LLM_API_KEY environment variable."
            )
        return self.llm.api_key

    def get_serpapi_key(self) -> Optional[str]:
        """Get SerpAPI key (optional)."""
        return self.api_keys.serpapi_key

    def get_doc2x_api_key(self) -> str:
        """Get Doc2X API key or raise error."""
        if not self.doc2x.api_key:
            raise ValueError(
                "Doc2X API key not configured. Set DOC2X_API_KEY environment variable "
                "or add [doc2x] api_key = '...' to your config file."
            )
        return self.doc2x.api_key

    def get_unpaywall_email(self) -> str:
        """Get Unpaywall email or raise error."""
        if not self.unpaywall.email:
            raise ValueError(
                "Unpaywall email not configured. Set UNPAYWALL_EMAIL environment variable "
                "or add [unpaywall] email = '...' to your config file. "
                "This is required for Unpaywall's polite pool access."
            )
        return self.unpaywall.email

    def get_ncbi_api_key(self) -> Optional[str]:
        """Get NCBI API key (optional, improves rate limits)."""
        return self.api_keys.ncbi_api_key

    def get_gemini_api_key(self) -> str:
        """Get Gemini API key or raise error."""
        if not self.gemini.api_key:
            raise ValueError(
                "Gemini API key not configured. Set GEMINI_API_KEY environment variable "
                "or add [gemini] api_key = '...' to your config file."
            )
        return self.gemini.api_key

    def get_zotero_api_key(self) -> Optional[str]:
        """Get Zotero API key (optional)."""
        return self.zotero.api_key

    def get_zotero_user_id(self) -> Optional[str]:
        """Get Zotero user ID (optional)."""
        return self.zotero.user_id

