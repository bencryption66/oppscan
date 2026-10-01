"""Etsy Open API v3 client: pacing, retries, a rolling daily quota and a per-run response cache."""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import timedelta

import httpx

from oppscan.db import utcnow

BASE_URL = "https://openapi.etsy.com/v3/application"
RETRYABLE = {429, 500, 502, 503, 504}


class QuotaExhausted(Exception):
    """The rolling 24-hour call budget is used up, or Etsy kept rate-limiting us."""


class EtsyError(Exception):
    """A non-retryable API failure, or retries exhausted."""

    def __init__(self, endpoint: str, status: int, body: str):
        super().__init__(f"{endpoint} -> HTTP {status}: {body[:200]}")
        self.endpoint = endpoint
        self.status = status


class EtsyClient:
    def __init__(self, con, run_id: str, *, api_key: str, qps: float, daily_quota: int,
                 transport: httpx.BaseTransport | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic,
                 max_retries: int = 5):
        self._con = con
        self._run_id = run_id
        self._daily_quota = daily_quota
        self._min_interval = 1.0 / qps
        self._sleep = sleep
        self._clock = clock
        self._max_retries = max_retries
        self._last_call: float | None = None
        self._http = httpx.Client(base_url=BASE_URL, headers={"x-api-key": api_key},
                                  transport=transport, timeout=30.0)

    def close(self) -> None:
        self._http.close()

    def get(self, endpoint: str, params: dict | None = None) -> dict:
        params = params or {}
        request_key = json.dumps(params, sort_keys=True)
        row = self._con.execute(
            "SELECT payload FROM raw_api WHERE run_id = ? AND endpoint = ? AND request_key = ?",
            [self._run_id, endpoint, request_key],
        ).fetchone()
        if row is not None:
            return json.loads(row[0])
        if self.calls_last_24h() >= self._daily_quota:
            raise QuotaExhausted(f"Etsy daily quota of {self._daily_quota} calls reached")
        payload = self._fetch(endpoint, params)
        self._con.execute("INSERT INTO raw_api VALUES (?, ?, ?, ?, ?)",
                          [self._run_id, endpoint, request_key, utcnow(), json.dumps(payload)])
        self._con.execute("UPDATE runs SET api_calls = api_calls + 1 WHERE run_id = ?", [self._run_id])
        return payload

    def calls_last_24h(self) -> int:
        return self._con.execute("SELECT count(*) FROM raw_api WHERE fetched_at >= ?",
                                 [utcnow() - timedelta(hours=24)]).fetchone()[0]

    def _pace(self) -> None:
        if self._last_call is not None:
            wait = self._min_interval - (self._clock() - self._last_call)
            if wait > 0:
                self._sleep(wait)
        self._last_call = self._clock()

    def _fetch(self, endpoint: str, params: dict) -> dict:
        for attempt in range(self._max_retries + 1):
            self._pace()
            try:
                response = self._http.get(endpoint, params=params)
            except httpx.TransportError as exc:
                if attempt < self._max_retries:
                    self._sleep(min(60.0, 2.0 ** attempt))
                    continue
                raise EtsyError(endpoint, 0, str(exc))
            if response.status_code == 200:
                return response.json()
            if response.status_code in RETRYABLE and attempt < self._max_retries:
                self._sleep(self._backoff(response, attempt))
                continue
            if response.status_code == 429:
                raise QuotaExhausted("Etsy rate limit persisted after retries")
            raise EtsyError(endpoint, response.status_code, response.text)
        raise AssertionError("unreachable")

    @staticmethod
    def _backoff(response: httpx.Response, attempt: int) -> float:
        retry_after = response.headers.get("retry-after", "")
        if retry_after.isdigit():
            return float(retry_after)
        return min(60.0, 2.0 ** attempt)
