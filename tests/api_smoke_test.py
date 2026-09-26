import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


def get_json(path: str) -> dict:
    request = Request(
        f"{BASE_URL}{path}",
        headers={"Accept": "application/json"},
    )

    try:
        with urlopen(request, timeout=5) as response:
            assert response.status == 200, f"{path}: expected 200, got {response.status}"
            return json.load(response)
    except HTTPError as exc:
        raise AssertionError(f"{path}: HTTP {exc.code}") from exc
    except URLError as exc:
        raise AssertionError(f"{path}: API unavailable: {exc.reason}") from exc


def run() -> None:
    live = get_json("/health/live")
    assert live == {"status": "ok"}, f"Unexpected live response: {live}"

    ready = get_json("/health/ready")
    assert ready.get("status") == "ok", f"API is not ready: {ready}"
    assert ready.get("database") == "sonar_mvp", f"Unexpected database: {ready}"
    assert isinstance(ready.get("postgis"), str) and ready["postgis"], (
        f"PostGIS version missing: {ready}"
    )

    jobs = get_json("/ingestion/jobs?page=1&page_size=10")
    assert isinstance(jobs.get("items"), list), f"items must be a list: {jobs}"
    assert isinstance(jobs.get("total"), int), f"total must be an integer: {jobs}"
    assert isinstance(jobs.get("page"), int) and jobs["page"] == 1, (
        f"Unexpected page: {jobs}"
    )
    assert isinstance(jobs.get("page_size"), int) and jobs["page_size"] == 10, (
        f"Unexpected page_size: {jobs}"
    )
    assert isinstance(jobs.get("pages"), int) and jobs["pages"] >= 0, (
        f"Unexpected pages: {jobs}"
    )
    assert jobs["total"] >= len(jobs["items"]), f"Invalid pagination totals: {jobs}"

    print("Live health endpoint: PASS")
    print("Readiness/database endpoint: PASS")
    print("Ingestion jobs API contract: PASS")
    print("API smoke test: PASS")


if __name__ == "__main__":
    run()
