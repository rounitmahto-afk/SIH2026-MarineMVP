import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


def request_json(path: str):
    request = Request(
        f"{BASE_URL}{path}",
        headers={"Accept": "application/json"},
    )

    try:
        with urlopen(request, timeout=5) as response:
            return response.status, json.load(response)
    except HTTPError as exc:
        body = exc.read().decode("utf-8")
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = body
        return exc.code, payload
    except URLError as exc:
        raise AssertionError(f"API unavailable: {exc.reason}") from exc


def assert_validation_error(path: str, message: str) -> None:
    status, payload = request_json(path)
    assert status == 422, f"{message}: expected 422, got {status} ({payload})"
    assert isinstance(payload, dict) and "detail" in payload, (
        f"{message}: validation error must include detail ({payload})"
    )


def run() -> None:
    status, payload = request_json("/ingestion/jobs?page=1&page_size=10")
    assert status == 200, f"Valid pagination request failed: {status} ({payload})"
    assert isinstance(payload, dict), f"Unexpected list response: {payload}"

    assert_validation_error(
        "/ingestion/jobs?page=0&page_size=10",
        "Page zero must be rejected",
    )

    assert_validation_error(
        "/ingestion/jobs?page=1&page_size=0",
        "Page size zero must be rejected",
    )

    assert_validation_error(
        "/ingestion/jobs?page=1&page_size=101",
        "Page size above 100 must be rejected",
    )

    status, payload = request_json("/ingestion/jobs/999999999")
    assert status == 404, f"Unknown job must return 404, got {status} ({payload})"
    assert payload == {"detail": "Ingestion job not found."}, (
        f"Unexpected unknown-job response: {payload}"
    )

    print("Valid ingestion pagination: PASS")
    print("Invalid page rejection: PASS")
    print("Invalid page_size lower bound: PASS")
    print("Invalid page_size upper bound: PASS")
    print("Unknown ingestion job rejection: PASS")
    print("Ingestion API edge-case tests: PASS")


if __name__ == "__main__":
    run()
