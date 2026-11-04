"""Actual HTTP boundary: deadlines, size bounds, redirects, and redaction."""

import json
import time

import pytest
import requests

from src.llm_client import HTTPTransport, ServiceFailure


class Response:
    def __init__(self, content, status=200):
        self.content, self.status_code = content, status
        self.headers = {"Retry-After": "60"}
        self.closed = False

    def iter_content(self, chunk_size):
        yield self.content

    def close(self):
        self.closed = True


def test_transport_only_uses_official_host_and_does_not_follow_redirects(monkeypatch):
    response = Response(b'{"data":{}}', 302)
    observed = {}
    def request(method, url, **kwargs):
        observed.update(method=method, url=url, **kwargs)
        return response
    monkeypatch.setattr(requests, "request", request)
    result = HTTPTransport().send("GET", "/key", "secret", None, time.monotonic() + 20)
    assert observed["url"] == "https://openrouter.ai/api/v1/key"
    assert observed["allow_redirects"] is False
    assert observed["headers"]["Authorization"] == "Bearer secret"
    assert result.status == 302 and result.headers == {"retry-after": "60"}
    assert response.closed


@pytest.mark.parametrize("exception,reason", [(requests.Timeout("secret"), "timeout"),
    (requests.ConnectionError("secret"), "connection_error")])
def test_network_errors_do_not_expose_requests_or_credentials(monkeypatch, exception, reason):
    def request(*args, **kwargs):
        raise exception
    monkeypatch.setattr(requests, "request", request)
    with pytest.raises(ServiceFailure) as caught:
        HTTPTransport().send("GET", "/key", "secret", None, time.monotonic() + 20)
    assert str(caught.value) == reason and "secret" not in str(caught.value)


@pytest.mark.parametrize("content,reason", [(b'x' * 80001, "response_too_large"), (b'not-json-secret', "invalid_response"),
    (b'{"value":NaN}', "invalid_response"), (b'[]', "invalid_response")])
def test_bad_response_is_bounded_closed_and_redacted(monkeypatch, content, reason):
    response = Response(content)
    monkeypatch.setattr(requests, "request", lambda *a, **kw: response)
    with pytest.raises(ServiceFailure) as caught:
        HTTPTransport().send("POST", "/chat/completions", "secret", {}, time.monotonic() + 20)
    assert str(caught.value) == reason and response.closed


def test_expired_deadline_never_sends_a_request(monkeypatch):
    monkeypatch.setattr(requests, "request", lambda *a, **kw: pytest.fail("Expired request should not connect"))
    with pytest.raises(ServiceFailure, match="time_budget"):
        HTTPTransport().send("GET", "/key", "secret", None, time.monotonic() - 1)
