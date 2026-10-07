"""Generate an API key for a tenant.

The clear-text key is printed once and must be handed to the client; only its SHA-256 digest
goes into the server configuration (API_KEYS).

    uv run python -m scripts.generate_api_key --tenant acme
"""

from __future__ import annotations

import argparse
import json

from app.core.config import ApiKeyEntry
from app.core.security import generate_api_key, hash_api_key


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant", required=True, help="tenant id (lowercase, digits, - and _)")
    parser.add_argument("--name", default="default", help="label of the key, e.g. 'ci' or 'alice'")
    args = parser.parse_args(argv)

    api_key = generate_api_key()
    entry = ApiKeyEntry(tenant_id=args.tenant, key_sha256=hash_api_key(api_key), name=args.name)
    print("API key (give it to the client, it is not stored anywhere):")
    print(f"  {api_key}")
    print("\nEntry to add to the API_KEYS JSON list:")
    print(f"  {json.dumps(entry.model_dump())}")


if __name__ == "__main__":
    main()
