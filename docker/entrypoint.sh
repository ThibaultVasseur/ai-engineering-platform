#!/bin/sh
# Container entrypoint: apply migrations (with retries while the database starts), optionally
# seed the demo knowledge base, then hand over to the command (uvicorn by default).
set -eu

if [ "${RUN_MIGRATIONS:-true}" = "true" ]; then
  attempt=1
  until alembic upgrade head; do
    if [ "$attempt" -ge 15 ]; then
      echo "entrypoint: migrations failed after $attempt attempts" >&2
      exit 1
    fi
    attempt=$((attempt + 1))
    echo "entrypoint: database not ready, retrying in 2s ($attempt/15)" >&2
    sleep 2
  done
fi

if [ "${SEED_DEMO_DATA:-false}" = "true" ]; then
  python -m scripts.seed_knowledge --tenant "${DEFAULT_TENANT_ID:-default}"
fi

exec "$@"
