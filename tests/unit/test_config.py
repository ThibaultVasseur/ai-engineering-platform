import pytest
from pydantic import ValidationError

from app.core.config import ApiKeyEntry, Settings

DIGEST = "a" * 64


def make_settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def test_defaults_run_offline_without_any_secret() -> None:
    settings = make_settings()
    assert settings.llm_provider == "offline"
    assert settings.effective_llm_model == "offline-simulator-v1"
    assert settings.embedding_provider == "hashing"
    assert settings.langfuse_enabled is False


def test_anthropic_defaults_to_current_opus_model() -> None:
    settings = make_settings(llm_provider="anthropic")
    assert settings.effective_llm_model == "claude-opus-5-5"


def test_explicit_model_overrides_provider_default() -> None:
    settings = make_settings(llm_provider="anthropic", llm_model="claude-sonnet-5-5")
    assert settings.effective_llm_model == "claude-sonnet-5-5"


def test_openai_provider_requires_explicit_model() -> None:
    with pytest.raises(ValidationError, match="LLM_MODEL must be set"):
        make_settings(llm_provider="openai")


def test_production_requires_authentication() -> None:
    with pytest.raises(ValidationError, match="requires AUTH_ENABLED"):
        make_settings(app_env="production")
    settings = make_settings(
        app_env="production",
        auth_enabled=True,
        api_keys=[ApiKeyEntry(tenant_id="acme", key_sha256=DIGEST)],
    )
    assert settings.auth_enabled


def test_chunk_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValidationError, match="RAG_CHUNK_OVERLAP"):
        make_settings(rag_chunk_size=300, rag_chunk_overlap=300)


def test_secrets_are_masked_in_repr() -> None:
    settings = make_settings(
        llm_api_key="sk-ant-should-never-appear", database_url="postgresql://u:pw@h/db"
    )
    rendered = repr(settings)
    assert "should-never-appear" not in rendered
    assert "pw@" not in rendered


def test_api_keys_are_parsed_from_json_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_KEYS", f'[{{"tenant_id": "acme", "key_sha256": "{DIGEST}"}}]')
    settings = make_settings()
    assert settings.api_keys == [ApiKeyEntry(tenant_id="acme", key_sha256=DIGEST)]


def test_empty_environment_values_mean_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("LLM_MODEL", "")
    settings = make_settings()
    assert settings.llm_api_key is None
    assert settings.llm_model is None


@pytest.mark.parametrize("tenant", ["Acme", "a b", "", "-start", "x" * 64])
def test_tenant_ids_are_restricted(tenant: str) -> None:
    with pytest.raises(ValidationError):
        ApiKeyEntry(tenant_id=tenant, key_sha256=DIGEST)
