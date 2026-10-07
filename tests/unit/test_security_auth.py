from app.core.config import ApiKeyEntry
from app.core.security import ApiKeyAuthenticator, Tenant, generate_api_key, hash_api_key


def test_generated_keys_are_unique_and_prefixed() -> None:
    keys = {generate_api_key() for _ in range(50)}
    assert len(keys) == 50
    assert all(key.startswith("aep_") and len(key) > 40 for key in keys)


def test_authenticate_maps_key_to_its_tenant() -> None:
    acme_key, globex_key = generate_api_key(), generate_api_key()
    authenticator = ApiKeyAuthenticator(
        [
            ApiKeyEntry(tenant_id="acme", key_sha256=hash_api_key(acme_key), name="ci"),
            ApiKeyEntry(tenant_id="globex", key_sha256=hash_api_key(globex_key)),
        ]
    )
    assert authenticator.authenticate(acme_key) == Tenant(id="acme", key_name="ci")
    assert authenticator.authenticate(globex_key) == Tenant(id="globex", key_name="default")


def test_unknown_missing_or_oversized_keys_are_rejected() -> None:
    authenticator = ApiKeyAuthenticator(
        [ApiKeyEntry(tenant_id="acme", key_sha256=hash_api_key(generate_api_key()))]
    )
    assert authenticator.authenticate(None) is None
    assert authenticator.authenticate("") is None
    assert authenticator.authenticate(generate_api_key()) is None
    assert authenticator.authenticate("x" * 10_000) is None


def test_configuration_only_contains_digests() -> None:
    key = generate_api_key()
    entry = ApiKeyEntry(tenant_id="acme", key_sha256=hash_api_key(key))
    assert key not in entry.model_dump_json()
