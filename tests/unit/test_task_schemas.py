import pytest
from pydantic import ValidationError

from app.schemas.task import MAX_REQUEST_CHARS, TaskCreate


def test_request_is_normalised() -> None:
    task = TaskCreate(request="  Build a quote​ system for a SaaS  ")
    assert task.request == "Build a quote system for a SaaS"


@pytest.mark.parametrize(
    "request_text",
    ["", "too short", "​" * 30, "x" * (MAX_REQUEST_CHARS + 1)],
)
def test_invalid_requests_are_rejected(request_text: str) -> None:
    with pytest.raises(ValidationError):
        TaskCreate(request=request_text)


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        TaskCreate.model_validate({"request": "A valid request text", "status": "COMPLETED"})


def test_metadata_is_bounded() -> None:
    TaskCreate(request="A valid request text", metadata={"source": "crm", "priority": 2})
    with pytest.raises(ValidationError):
        TaskCreate(request="A valid request text", metadata={f"k{i}": i for i in range(21)})
    with pytest.raises(ValidationError):
        TaskCreate(request="A valid request text", metadata={"note": "x" * 501})
