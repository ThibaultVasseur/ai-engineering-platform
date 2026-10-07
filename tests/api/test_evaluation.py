import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.integration


async def test_evaluation_runs_in_a_sandbox_and_is_persisted(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/evaluation/run",
        json={"case_ids": ["rag-quote-validity", "injection-system-prompt"]},
    )
    assert response.status_code == 200
    evaluation = response.json()
    assert evaluation["status"] == "COMPLETED"
    assert evaluation["summary"]["cases"] == 2
    assert evaluation["summary"]["success_rate"] == 1.0
    assert {r["id"] for r in evaluation["results"]} == {
        "rag-quote-validity",
        "injection-system-prompt",
    }

    fetched = (await client.get(f"/api/v1/evaluation/{evaluation['id']}")).json()
    assert fetched["id"] == evaluation["id"]
    listed = (await client.get("/api/v1/evaluation")).json()
    assert [e["id"] for e in listed] == [evaluation["id"]]
    # The sandbox never writes into the tenant's knowledge base.
    assert (await client.get("/api/v1/knowledge/documents")).json()["items"] == []


async def test_unknown_dataset_or_case_is_rejected(client: AsyncClient) -> None:
    assert (
        await client.post("/api/v1/evaluation/run", json={"dataset": "nope"})
    ).status_code == 422
    unknown_case = await client.post("/api/v1/evaluation/run", json={"case_ids": ["ghost"]})
    assert unknown_case.status_code == 422
