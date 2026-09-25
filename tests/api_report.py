import json
import unittest
from contextlib import contextmanager
from pathlib import Path


OUTPUT_PATH = Path(__file__).resolve().parent / "outputs" / "control_api.md"
_CASES: list[dict] = []
_CURRENT = "setup"
_INSTALLED = False


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    from fastapi.testclient import TestClient

    result_cls = unittest.TextTestResult
    start_test = result_cls.startTest
    stop_test_run = result_cls.stopTestRun
    original_request = TestClient.request

    def remember_start(self, test):
        global _CURRENT
        _CURRENT = test.id().rsplit(".", 1)[-1]
        return start_test(self, test)

    def remember_stop(self):
        try:
            return stop_test_run(self)
        finally:
            write()

    def request(self, method, url, *args, **kwargs):
        response = original_request(self, method, url, *args, **kwargs)
        remember(_CURRENT, method, url, response)
        return response

    result_cls.startTest = remember_start
    result_cls.stopTestRun = remember_stop
    TestClient.request = request


def remember(test_name: str, method: str, url: str, response) -> None:
    path = str(url).split("?", 1)[0]
    if path.startswith("http://"):
        path = "/" + path.split("/", 3)[-1]
    try:
        body = response.json()
    except Exception:
        body = response.text
    _CASES.append(
        {
            "test": test_name,
            "method": method.upper(),
            "path": path,
            "status_code": response.status_code,
            "body": body,
        }
    )


@contextmanager
def recorded_client(app, test_name: str):
    global _CURRENT
    previous = _CURRENT
    _CURRENT = test_name
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        try:
            yield client
        finally:
            _CURRENT = previous


def write() -> None:
    if not _CASES:
        return
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Control API 테스트 결과",
        "",
        "fake worker로 control API와 전사 API를 호출한 기록이다.",
        "",
    ]
    current = None
    for case in _CASES:
        if case["test"] != current:
            current = case["test"]
            lines.extend([f"## {current}", ""])
        lines.extend(
            [
                f"### {case['method']} {case['path']}",
                "",
                f"HTTP {case['status_code']}",
                "",
                "```json",
                json.dumps(case["body"], ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    stale = OUTPUT_PATH.with_name("api-tests.md")
    if stale.is_file():
        stale.unlink()
