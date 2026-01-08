"""Configuration management for papercli."""

import os
from pathlib import Path
from typing import Optional

import toml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """LLM configuration."""

    base_url: str = Field(default="https://api.openai.com/v1")
    api_key: Optional[str] = Field(default=None)
    intent_model: str = Field(default="gpt-4o-mini")
    eval_model: str = Field(default="gpt-4o")
    timeout: float = Field(default=60.0)
    max_retries: int = Field(default=3)


class CacheConfig(BaseModel):
    """Cache configuration."""

    path: str = Field(default="~/.cache/papercli.sqlite")
    enabled: bool = Field(default=True)
    ttl_hours: int = Field(default=24 * 7)  # 1 week default


class APIKeysConfig(BaseModel):
    """API keys configuration."""

    serpapi_key: Optional[str] = Field(default=None)
    ncbi_api_key: Optional[str] = Field(default=None)  # Optional for PubMed
    openalex_email: Optional[str] = Field(default=None)  # Polite pool for OpenAlex


class Settings(BaseModel):
    """Main settings container."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    api_keys: APIKeysConfig = Field(default_factory=APIKeysConfig)

    # CLI overrides (set directly from command line)
    intent_model: Optional[str] = Field(default=None)
    eval_model: Optional[str] = Field(default=None)
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
        if intent_model := os.environ.get("PAPERCLI_INTENT_MODEL"):
            self.llm.intent_model = intent_model
        if eval_model := os.environ.get("PAPERCLI_EVAL_MODEL"):
            self.llm.eval_model = eval_model

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
                    break
                except Exception:
                    pass  # Ignore invalid config files

    def _merge_config(self, config_data: dict) -> None:
        """Merge config file data into settings."""
        if llm_config := config_data.get("llm"):
            if "base_url" in llm_config and not self.llm_base_url:
                self.llm.base_url = llm_config["base_url"]
            if "intent_model" in llm_config and not self.intent_model:
                self.llm.intent_model = llm_config["intent_model"]
            if "eval_model" in llm_config and not self.eval_model:
                self.llm.eval_model = llm_config["eval_model"]
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

    def _apply_cli_overrides(self) -> None:
        """Apply CLI-provided overrides."""
        if self.intent_model:
            self.llm.intent_model = self.intent_model
        if self.eval_model:
            self.llm.eval_model = self.eval_model
        if self.llm_base_url:
            self.llm.base_url = self.llm_base_url
        if self.cache_path:
            self.cache.path = self.cache_path
        if not self.cache_enabled:
            self.cache.enabled = False

    def get_intent_model(self) -> str:
        """Get the model to use for intent extraction."""
        return self.intent_model or self.llm.intent_model

    def get_eval_model(self) -> str:
        """Get the model to use for evaluation/reranking."""
        return self.eval_model or self.llm.eval_model

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

