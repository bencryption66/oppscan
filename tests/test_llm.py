import json
from types import SimpleNamespace

import pytest

from oppscan.fake_llm import FakeLLM
from oppscan.llm import AnthropicLLM, LLMOutputError, validate
from oppscan.prompts import SCHEMAS, render

GOOD_CLUSTER = {"niches": [{"name": "Budget", "description": "d", "seeds": ["budget spreadsheet"]}]}
PAYLOAD = {"seeds": [{"seed": "budget spreadsheet", "titles": ["Budget Spreadsheet"]}]}


def message(text, stop_reason="end_turn"):
    return SimpleNamespace(stop_reason=stop_reason, content=[SimpleNamespace(type="text", text=text)])


class FakeAnthropic:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return self.responses.pop(0)


def test_schemas_are_strict():
    def check(schema):
        if schema.get("type") == "object":
            assert schema["additionalProperties"] is False
            assert set(schema["required"]) == set(schema["properties"])
            for sub in schema["properties"].values():
                check(sub)
        if schema.get("type") == "array":
            check(schema["items"])

    for schema in SCHEMAS.values():
        check(schema)


def test_render_embeds_payload():
    system, user = render("cluster", PAYLOAD)
    assert "Etsy" in system
    assert '"budget spreadsheet"' in user


def test_validate_rejects_bad_shape():
    with pytest.raises(LLMOutputError):
        validate("cluster", {"niches": [{"name": "x"}]})


def test_anthropic_llm_returns_and_caches(con):
    client = FakeAnthropic(message(json.dumps(GOOD_CLUSTER)))
    llm = AnthropicLLM(con, client=client)
    assert llm.call("cluster", PAYLOAD) == GOOD_CLUSTER
    assert llm.call("cluster", PAYLOAD) == GOOD_CLUSTER
    assert len(client.requests) == 1
    request = client.requests[0]
    assert request["model"] == "claude-sonnet-5-5"
    assert request["output_config"]["format"]["type"] == "json_schema"
    assert request["output_config"]["format"]["schema"] == SCHEMAS["cluster"]
    assert request["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in request["betas"]


def test_anthropic_llm_retries_once_on_invalid_output(con):
    client = FakeAnthropic(message("not json"), message(json.dumps(GOOD_CLUSTER)))
    assert AnthropicLLM(con, client=client).call("cluster", PAYLOAD) == GOOD_CLUSTER
    assert len(client.requests) == 2


def test_anthropic_llm_raises_after_two_failures(con):
    client = FakeAnthropic(message('{"niches": [{}]}'), message("still bad"))
    with pytest.raises(LLMOutputError):
        AnthropicLLM(con, client=client).call("cluster", PAYLOAD)


def test_anthropic_llm_treats_refusal_as_failure(con):
    client = FakeAnthropic(message("", "refusal"), message("", "refusal"))
    with pytest.raises(LLMOutputError, match="refusal"):
        AnthropicLLM(con, client=client).call("cluster", PAYLOAD)


def test_fake_llm_outputs_validate():
    fake = FakeLLM()
    clusters = fake.call("cluster", {"seeds": [{"seed": "budget a", "titles": []},
                                               {"seed": "budget b", "titles": []},
                                               {"seed": "notion c", "titles": []}]})
    assert [n["seeds"] for n in clusters["niches"]] == [["budget a", "budget b"], ["notion c"]]
    themes = fake.call("complaints", {"niche": "x", "reviews": [{"id": "a", "rating": 2, "text": "bad"},
                                                                {"id": "b", "rating": 5, "text": "wish"}]})
    assert themes["themes"][0]["review_ids"] == ["a"]
    brief = fake.call("brief", {"niche": {"name": "Budget", "description": "d"},
                                "metrics": {"median_price": 12.5}, "complaints": [], "competitors": []})
    assert brief["suggested_price_usd"] == 12.5
    assert fake.call("summary", {"top": [], "dropped": 0, "suspect": False, "status_reasons": []})["summary"]
    assert [t for t, _ in fake.calls] == ["cluster", "complaints", "brief", "summary"]
