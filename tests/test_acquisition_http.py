"""Exercise real loopback HTTP transport, not mocked HTTP response objects."""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Thread
from time import sleep
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from src.acquisition import AcquisitionError, AcquisitionSession, InMemoryTestAdapter, LocalHTTPAdapter, test_source_records as source_records
from src.data_generation import SimulationConfig, generate_dataset
from src.data_integration import sources_from_dataset
from src.test_acquisition_source import make_test_server


@contextmanager
def serving(server):
    thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.fixture(scope="module")
def adapter():
    sources = sources_from_dataset(generate_dataset(SimulationConfig(n_batches=1, duration_h=12, n_wavenumbers=32)))
    return InMemoryTestAdapter(source_records(sources))


def test_http_adapter_matches_in_app_contract_and_resumes_from_cursor(adapter):
    with serving(make_test_server(adapter)) as base:
        http = LocalHTTPAdapter(base)
        first = http.poll(0, 5)
        assert first == adapter.poll(0, 5)
        session = AcquisitionSession("http")
        assert session.poll_once(http, 5)
        assert session.poll_once(http, 5)
        assert session.cursor == 10 and len(session.audit) == 10
        with urlopen(base + "/health", timeout=2) as response:
            assert json.load(response)["source_id"] == adapter.source_id


def test_standalone_test_server_is_read_only_and_checks_query_contract(adapter):
    with serving(make_test_server(adapter)) as base:
        for path, code in (("/v1/records?after=0&limit=0", 400), ("/v1/records?after=0&after=1&limit=5", 400), ("/other", 404)):
            with pytest.raises(HTTPError) as error:
                urlopen(base + path, timeout=2)
            assert error.value.code == code
        with pytest.raises(HTTPError) as error:
            urlopen(Request(base + "/v1/records", data=b"{}", method="POST"), timeout=2)
        assert error.value.code == 405


@contextmanager
def bad_source(body=b"{}", *, content_type="application/json", status=200, delay=0, redirect=None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if delay:
                sleep(delay)
            try:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                if redirect:
                    self.send_header("Location", redirect)
                self.end_headers()
                self.wfile.write(body)
            except OSError:
                pass  # Client deliberately timed out or rejected a large response.
    with serving(ThreadingHTTPServer(("127.0.0.1", 0), Handler)) as base:
        yield base


@pytest.mark.parametrize("body", [b"not-json", b'{"schema_version":1,"schema_version":1}', b'{"schema_version":true}', b'{"schema_version":99}', b'{"schema_version":1,"source_id":NaN}'])
def test_invalid_json_and_schema_versions_leave_collection_unchanged(body):
    with bad_source(body) as base:
        session = AcquisitionSession("malformed")
        assert not session.poll_once(LocalHTTPAdapter(base), 5)
        assert session.cursor == 0 and not session.audit and not session.running


def test_wrong_content_type_server_error_and_redirect_are_rejected():
    for options in ({"content_type": "text/html"}, {"status": 503}, {"status": 302, "redirect": "http://example.com/"}):
        with bad_source(**options) as base:
            with pytest.raises(AcquisitionError):
                LocalHTTPAdapter(base).poll(0, 5)


def test_timeout_is_reported_without_cursor_progress():
    with bad_source(delay=0.15) as base:
        session = AcquisitionSession("timeout")
        session.running = True
        assert not session.poll_once(LocalHTTPAdapter(base, timeout_seconds=0.02), 5)
        assert session.cursor == 0 and not session.running and session.last_error


def test_oversized_response_is_rejected_before_parsing():
    with bad_source(b" " * (2 * 1024 * 1024 + 1)) as base:
        with pytest.raises(AcquisitionError, match="2 MiB"):
            LocalHTTPAdapter(base).poll(0, 5)


def test_stopped_server_read_is_recoverable_and_retains_prior_rows(adapter):
    with serving(make_test_server(adapter)) as base:
        session = AcquisitionSession("disconnect")
        assert session.poll_once(LocalHTTPAdapter(base), 5)
    assert not session.poll_once(LocalHTTPAdapter(base, timeout_seconds=0.1), 5)
    assert session.cursor == 5 and len(session.audit) == 5
