import uuid

import pytest
from httpx import AsyncClient

from app.core.config import ApiKeyEntry
from app.core.security import generate_api_key, hash_api_key
from app.db.database import Database
from tests.support import AppClient, make_settings

pytestmark = pytest.mark.integration

REQUEST = "Analyse les documents et propose une architecture de gestion de devis."


async def test_create_then_get_task(client: AsyncClient) -> None:
    created = await client.post("/api/v1/tasks", json={"request": REQUEST, "metadata": {"a": 1}})
    assert created.status_code == 201
    body = created.json()
    assert body["status"] == "PENDING"
    assert body["user_request"] == REQUEST
    assert body["metadata"] == {"a": 1}

    fetched = await client.get(f"/api/v1/tasks/{body['id']}")
    assert fetched.status_code == 200
    assert fetched.json()["id"] == body["id"]


async def test_list_tasks_most_recent_first_and_filter(client: AsyncClient) -> None:
    for index in range(3):
        await client.post("/api/v1/tasks", json={"request": f"{REQUEST} #{index}"})
    page = (await client.get("/api/v1/tasks", params={"limit": 2})).json()
    assert len(page["items"]) == 2
    assert page["items"][0]["user_request"].endswith("#2")
    done = (await client.get("/api/v1/tasks", params={"status": "COMPLETED"})).json()
    assert done["items"] == []


async def test_validation_errors_use_the_error_envelope(client: AsyncClient) -> None:
    response = await client.post("/api/v1/tasks", json={"request": "short"})
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert error["request_id"]
    assert "input" not in error["details"][0]  # rejected input is never echoed back


async def test_unknown_fields_are_rejected(client: AsyncClient) -> None:
    response = await client.post("/api/v1/tasks", json={"request": REQUEST, "status": "COMPLETED"})
    assert response.status_code == 422


async def test_missing_task_is_404(client: AsyncClient) -> None:
    response = await client.get(f"/api/v1/tasks/{uuid.uuid4()}")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


async def test_tenants_cannot_see_each_other(database: Database) -> None:
    acme_key, globex_key = generate_api_key(), generate_api_key()
    settings = make_settings(
        auth_enabled=True,
        api_keys=[
            ApiKeyEntry(tenant_id="acme", key_sha256=hash_api_key(acme_key)),
            ApiKeyEntry(tenant_id="globex", key_sha256=hash_api_key(globex_key)),
        ],
    )
    async with AppClient(settings) as client:
        unauthenticated = await client.post("/api/v1/tasks", json={"request": REQUEST})
        assert unauthenticated.status_code == 401

        created = await client.post(
            "/api/v1/tasks", json={"request": REQUEST}, headers={"X-API-Key": acme_key}
        )
        task_id = created.json()["id"]

        as_globex = await client.get(f"/api/v1/tasks/{task_id}", headers={"X-API-Key": globex_key})
        assert as_globex.status_code == 404
        globex_list = await client.get("/api/v1/tasks", headers={"X-API-Key": globex_key})
        assert globex_list.json()["items"] == []

        as_acme = await client.get(f"/api/v1/tasks/{task_id}", headers={"X-API-Key": acme_key})
        assert as_acme.status_code == 200
