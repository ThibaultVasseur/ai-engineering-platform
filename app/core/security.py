"""Authentication and tenant resolution.

API keys are random 256-bit tokens. The server only stores (in configuration) their SHA-256
digest, so a leaked configuration does not leak usable credentials. Every key maps to exactly
one tenant; the tenant id then scopes every database query (and PostgreSQL row-level security).
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass

from app.core.config import ApiKeyEntry

API_KEY_PREFIX = "aep_"


@dataclass(frozen=True, slots=True)
class Tenant:
    id: str
    key_name: str | None = None


def generate_api_key() -> str:
    return API_KEY_PREFIX + secrets.token_urlsafe(32)


def hash_api_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


class ApiKeyAuthenticator:
    def __init__(self, entries: list[ApiKeyEntry]) -> None:
        self._entries = list(entries)

    def authenticate(self, presented_key: str | None) -> Tenant | None:
        if not presented_key or len(presented_key) > 256:
            return None
        digest = hash_api_key(presented_key)
        match: ApiKeyEntry | None = None
        # Compare against every entry (no early exit) in constant time.
        for entry in self._entries:
            if hmac.compare_digest(digest, entry.key_sha256):
                match = entry
        if match is None:
            return None
        return Tenant(id=match.tenant_id, key_name=match.name)
