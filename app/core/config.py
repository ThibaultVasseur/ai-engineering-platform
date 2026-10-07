"""Centralised, typed configuration.

Every setting is read from the environment (or a local ``.env`` file) and validated once at
start-up. Secrets are ``SecretStr`` so they never leak through ``repr()``, tracebacks or logs.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Final, Literal

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProviderName = Literal["offline", "anthropic", "openai"]
EmbeddingProviderName = Literal["hashing", "openai"]
Effort = Literal["low", "medium", "high", "xhigh", "max"]
SupervisorStrategy = Literal["rules", "llm"]

TENANT_ID_PATTERN = r"^[a-z0-9][a-z0-9_-]{0,62}$"

# Fixed by migration 0001 (vector(512) column). Changing it requires a new migration and a
# re-embedding of the knowledge base, so it is deliberately not an environment variable.
EMBEDDING_DIMENSIONS: Final = 512

# Defaults per provider. OpenAI-compatible endpoints serve too many models to pick one
# safely, so LLM_MODEL is mandatory for that provider.
DEFAULT_MODELS: dict[str, str] = {
    "offline": "offline-simulator-v1",
    "anthropic": "claude-opus-5-5",
}


class ApiKeyEntry(BaseModel):
    """An API key maps to exactly one tenant. Only its SHA-256 digest is configured."""

    tenant_id: str = Field(pattern=TENANT_ID_PATTERN)
    key_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    name: str = "default"


class Settings(BaseSettings):
    # env_ignore_empty: "LLM_API_KEY=" in a .env means "not set", not "empty secret".
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_ignore_empty=True, extra="ignore"
    )

    # --- Application -------------------------------------------------------------------
    app_name: str = "ai-engineering-platform"
    app_env: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "console"] = "json"

    # --- Database ----------------------------------------------------------------------
    database_url: SecretStr = SecretStr("postgresql+asyncpg://app:app@localhost:5433/ai_platform")
    db_pool_size: int = Field(default=5, ge=1, le=50)
    db_echo: bool = False

    # --- Authentication & tenancy ------------------------------------------------------
    auth_enabled: bool = False
    api_keys: list[ApiKeyEntry] = Field(default_factory=list)
    default_tenant_id: str = Field(default="default", pattern=TENANT_ID_PATTERN)

    # --- Rate limiting and request size -------------------------------------------------
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = Field(default=60, ge=1)
    max_request_body_bytes: int = Field(default=1_048_576, ge=1_024)

    # --- LLM ---------------------------------------------------------------------------
    llm_provider: LLMProviderName = "offline"
    llm_model: str | None = None
    llm_api_key: SecretStr | None = None
    llm_base_url: str | None = None
    llm_effort: Effort = "medium"
    llm_max_output_tokens: int = Field(default=16_000, ge=256, le=128_000)
    llm_timeout_seconds: float = Field(default=120.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0, le=10)
    llm_refusal_fallback: bool = True
    llm_structured_max_attempts: int = Field(default=3, ge=1, le=5)
    # Optional price override (USD per million tokens) for models missing from the price book.
    llm_price_input_per_mtok: float | None = Field(default=None, ge=0)
    llm_price_output_per_mtok: float | None = Field(default=None, ge=0)

    # --- Embeddings --------------------------------------------------------------------
    embedding_provider: EmbeddingProviderName = "hashing"
    embedding_model: str = "text-embedding-3-small"
    embedding_api_key: SecretStr | None = None
    embedding_base_url: str | None = None

    # --- Observability -----------------------------------------------------------------
    langfuse_host: str | None = None
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None

    # --- Orchestration limits ----------------------------------------------------------
    max_agent_steps: int = Field(default=40, ge=5, le=500)
    max_agent_calls: int = Field(default=16, ge=1, le=200)
    max_agent_retries: int = Field(default=2, ge=0, le=10)
    max_step_attempts: int = Field(default=2, ge=1, le=5)
    max_tool_calls_per_agent: int = Field(default=6, ge=0, le=50)
    max_plan_steps: int = Field(default=6, ge=1, le=20)
    max_run_seconds: float = Field(default=300.0, gt=0)
    max_run_cost_usd: float = Field(default=2.0, gt=0)
    max_concurrent_runs: int = Field(default=4, ge=1, le=64)
    critic_pass_threshold: int = Field(default=70, ge=0, le=100)
    supervisor_strategy: SupervisorStrategy = "rules"
    tool_timeout_seconds: float = Field(default=10.0, gt=0)
    tool_max_output_chars: int = Field(default=8_000, ge=500)

    # --- RAG ---------------------------------------------------------------------------
    rag_chunk_size: int = Field(default=900, ge=200, le=8_000)
    rag_chunk_overlap: int = Field(default=150, ge=0)
    rag_top_k: int = Field(default=5, ge=1, le=20)
    rag_candidate_pool: int = Field(default=20, ge=1, le=200)
    rag_block_suspicious_chunks: bool = True

    @property
    def effective_llm_model(self) -> str:
        if self.llm_model:
            return self.llm_model
        return DEFAULT_MODELS[self.llm_provider]

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    @model_validator(mode="after")
    def _check_consistency(self) -> Settings:
        if self.rag_chunk_overlap >= self.rag_chunk_size:
            raise ValueError("RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE")
        if self.rag_candidate_pool < self.rag_top_k:
            raise ValueError("RAG_CANDIDATE_POOL must be >= RAG_TOP_K")
        if self.llm_provider == "openai" and not self.llm_model:
            raise ValueError("LLM_MODEL must be set explicitly when LLM_PROVIDER=openai")
        if (self.llm_price_input_per_mtok is None) != (self.llm_price_output_per_mtok is None):
            raise ValueError(
                "Set both LLM_PRICE_INPUT_PER_MTOK and LLM_PRICE_OUTPUT_PER_MTOK, or neither"
            )
        if self.app_env == "production" and not (self.auth_enabled and self.api_keys):
            raise ValueError(
                "APP_ENV=production requires AUTH_ENABLED=true and at least one API key"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    """Process-wide settings (validated once). Tests build their own ``Settings`` instead."""
    return Settings()
