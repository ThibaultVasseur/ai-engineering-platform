#!/usr/bin/env bash
# Runs once, as the PostgreSQL superuser, when the data volume is first created
# (/docker-entrypoint-initdb.d). The CI workflow runs it against its service container too.
#
# * creates the application role WITHOUT superuser / BYPASSRLS, so row-level security applies;
# * creates the application and test databases owned by that role;
# * installs pgvector (CREATE EXTENSION needs elevated privileges).
set -euo pipefail

APP_DB_USER="${APP_DB_USER:-app}"
APP_DB_PASSWORD="${APP_DB_PASSWORD:-app}"
APP_DB_NAME="${APP_DB_NAME:-ai_platform}"
SUPERUSER="${POSTGRES_USER:-postgres}"

psql -v ON_ERROR_STOP=1 --username "$SUPERUSER" --dbname postgres \
  -v app_user="$APP_DB_USER" -v app_password="$APP_DB_PASSWORD" <<'EOSQL'
SELECT format('CREATE ROLE %I LOGIN PASSWORD %L NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS',
              :'app_user', :'app_password')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = :'app_user') \gexec
EOSQL

for database in "$APP_DB_NAME" "${APP_DB_NAME}_test"; do
  psql -v ON_ERROR_STOP=1 --username "$SUPERUSER" --dbname postgres \
    -v db="$database" -v app_user="$APP_DB_USER" <<'EOSQL'
SELECT format('CREATE DATABASE %I OWNER %I', :'db', :'app_user')
WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = :'db') \gexec
EOSQL
  psql -v ON_ERROR_STOP=1 --username "$SUPERUSER" --dbname "$database" \
    -c "CREATE EXTENSION IF NOT EXISTS vector;"
done

echo "init.sh: role '$APP_DB_USER' and databases '$APP_DB_NAME', '${APP_DB_NAME}_test' ready"
