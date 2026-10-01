from datetime import datetime

import httpx
import pytest

from oppscan import db
from oppscan.etsy import EtsyClient, EtsyError, QuotaExhausted

T = datetime(2026, 10, 1)


def make_client(con, handler, **kw):
    run_id = db.start_run(con, "h", T)
    sleeps: list[float] = []
    client = EtsyClient(
        con, run_id, api_key="key123",
        qps=kw.pop("qps", 1000.0), daily_quota=kw.pop("daily_quota", 100),
        transport=httpx.MockTransport(handler), sleep=sleeps.append, **kw,
    )
    return client, run_id, sleeps


def test_get_stores_and_reuses_cached_response(con):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"count": 1, "results": [{"listing_id": 1}]})

    client, run_id, _ = make_client(con, handler)
    first = client.get("/listings/active", {"keywords": "budget", "offset": 0})
    second = client.get("/listings/active", {"offset": 0, "keywords": "budget"})
    assert first == second == {"count": 1, "results": [{"listing_id": 1}]}
    assert len(calls) == 1
    assert calls[0].url.path == "/v3/application/listings/active"
    assert calls[0].headers["x-api-key"] == "key123"
    assert db.get_run(con, run_id)["api_calls"] == 1


def test_retries_on_429_then_succeeds(con):
    responses = iter([httpx.Response(429), httpx.Response(200, json={"ok": True})])
    client, _, sleeps = make_client(con, lambda r: next(responses))
    assert client.get("/shops/1") == {"ok": True}
    assert 1.0 in sleeps


def test_uses_retry_after_header(con):
    responses = iter([httpx.Response(503, headers={"retry-after": "7"}), httpx.Response(200, json={})])
    client, _, sleeps = make_client(con, lambda r: next(responses))
    client.get("/shops/1")
    assert 7.0 in sleeps


def test_gives_up_after_max_retries(con):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(503, text="down")

    client, _, _ = make_client(con, handler, max_retries=2)
    with pytest.raises(EtsyError) as err:
        client.get("/shops/1")
    assert err.value.status == 503
    assert len(calls) == 3


def test_non_retryable_status_raises_immediately(con):
    client, _, _ = make_client(con, lambda r: httpx.Response(404, text="nope"))
    with pytest.raises(EtsyError) as err:
        client.get("/shops/1")
    assert err.value.status == 404


def test_quota_exhausted(con):
    client, _, _ = make_client(con, lambda r: httpx.Response(200, json={}), daily_quota=1)
    client.get("/shops/1")
    with pytest.raises(QuotaExhausted):
        client.get("/shops/2")


def test_cached_reads_do_not_count_against_quota(con):
    client, _, _ = make_client(con, lambda r: httpx.Response(200, json={}), daily_quota=1)
    client.get("/shops/1")
    client.get("/shops/1")  # cached, no QuotaExhausted


def test_paces_calls(con):
    client, _, sleeps = make_client(con, lambda r: httpx.Response(200, json={}), qps=4.0,
                                    clock=lambda: 0.0)
    client.get("/shops/1")
    client.get("/shops/2")
    assert sleeps == [0.25]
