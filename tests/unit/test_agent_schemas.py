"""Worker output contracts: format-level variations are normalised, meaningful errors rejected."""

import pytest
from pydantic import ValidationError

from app.schemas.agent import CodeFile, Endpoint, Finding


def test_source_labels_copied_in_citation_format_are_normalised() -> None:
    finding = Finding(claim="Quotes expire", evidence="policy", source_ids=["[S1]", " s2 "])
    assert finding.source_ids == ["S1", "S2"]


def test_source_labels_that_are_not_labels_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Finding(claim="Quotes expire", evidence="policy", source_ids=["source 1"])


@pytest.mark.parametrize(
    ("written", "expected"),
    [
        ("GET /api/v1/quotes?status=sent", "/api/v1/quotes"),
        ("api/v1/quotes/{quote_id}", "/api/v1/quotes/{quote_id}"),
        ("/api/v1/quotes/{quote_id}:send", "/api/v1/quotes/{quote_id}:send"),
    ],
)
def test_endpoint_paths_keep_only_the_path(written: str, expected: str) -> None:
    assert Endpoint(method="GET", path=written, purpose="List quotes").path == expected


def test_endpoint_paths_with_invalid_characters_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Endpoint(method="GET", path="/api/v1/quotes/<id>", purpose="Read a quote")


@pytest.mark.parametrize(
    ("written", "expected"),
    [
        ("./src/quotes/service.py", "src/quotes/service.py"),
        ("/src/quotes/service.py", "src/quotes/service.py"),
        ("src\\quotes\\service.py", "src/quotes/service.py"),
    ],
)
def test_code_file_paths_are_made_relative(written: str, expected: str) -> None:
    file = CodeFile(path=written, language="python", purpose="Service", content="pass")
    assert file.path == expected


@pytest.mark.parametrize("path", ["../etc/passwd", "src/../../secrets.txt", "C:/Windows/x.py"])
def test_code_file_paths_cannot_escape_or_be_absolute(path: str) -> None:
    with pytest.raises(ValidationError):
        CodeFile(path=path, language="python", purpose="Service", content="pass")
