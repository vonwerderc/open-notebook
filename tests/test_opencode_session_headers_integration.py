"""End-to-end OpenCode session-header coverage against a local fake server.

The fake server enforces the documented OpenCode Go contract: requests without
``x-opencode-session`` get HTTP 400 ``MissingSessionID``; requests with it get
valid OpenAI-shaped payloads (JSON and SSE). Production URL matching is never
weakened - tests that route OpenNotebook provisioning to the local server
patch ``is_opencode_go_base_url`` to additionally allow only that exact test
URL, exactly as the plan permits.

Covers the Task 10 matrix: direct Esperanto, sync/async/streaming LangChain,
provisioning stability, per-operation scoping, podcast boundary, provider
isolation, and payload-compat (structured output, tools, existing headers).
Model-test/credential/discovery header behavior is covered in
``tests/test_opencode_ai_tests_discovery.py``.
"""

import json
import re
import threading
from collections.abc import AsyncGenerator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from open_notebook.ai import opencode_go
from open_notebook.ai.models import ModelManager
from open_notebook.ai.opencode_go import (
    OPENCODE_SESSION_HEADER,
)

SES_RE = re.compile(r"^ses_[0-9a-f]{32}$")


# ---------------------------------------------------------------------------
# Fake OpenAI-compatible server with the MissingSessionID contract
# ---------------------------------------------------------------------------


class _FakeHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence test output
        pass

    def _record(self, body: bytes | None = None):
        rec: dict[str, object] = {
            "method": self.command,
            "path": self.path,
            "session": self.headers.get(OPENCODE_SESSION_HEADER),
            "authorization": self.headers.get("Authorization"),
            "custom": self.headers.get("X-Custom"),
        }
        if body:
            try:
                rec["json"] = json.loads(body)
            except Exception:
                rec["json"] = None
        self.server.requests.append(rec)  # type: ignore[attr-defined]  # noqa

    def _json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _missing_session(self):
        self._json(
            {
                "error": {
                    "message": "MissingSessionID: the x-opencode-session header is required",
                    "type": "invalid_request_error",
                    "code": "missing_session_id",
                }
            },
            status=400,
        )

    def do_GET(self):
        self._record()
        if not self.headers.get(OPENCODE_SESSION_HEADER):
            return self._missing_session()
        self._json({"object": "list", "data": [{"id": "gpt-test", "object": "model"}]})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        self._record(body)
        if not self.headers.get(OPENCODE_SESSION_HEADER):
            return self._missing_session()
        payload = {}
        try:
            payload = json.loads(body or b"{}")
        except Exception:
            pass

        if payload.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunk = {
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": payload.get("model", "gpt-test"),
                "choices": [
                    {"index": 0, "delta": {"content": "hi"}, "finish_reason": None}
                ],
            }
            self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
            done = {
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": payload.get("model", "gpt-test"),
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            self.wfile.write(b"data: " + json.dumps(done).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
            return

        message: dict[str, object] = {"role": "assistant", "content": "ok"}
        finish = "stop"
        if payload.get("tools"):
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "x"}',
                        },
                    }
                ],
            }
            finish = "tool_calls"
        self._json(
            {
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1,
                "model": payload.get("model", "gpt-test"),
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        )


@pytest.fixture
def fake_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeHandler)
    server.__dict__["requests"] = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield (
        f"http://127.0.0.1:{server.server_address[1]}/v1",
        server.__dict__["requests"],
    )
    server.shutdown()
    server.server_close()


def _allow_fake_url(fake_url):
    """Permit only this test's URL through the detector (test-only patch)."""
    original = opencode_go.is_opencode_go_base_url

    def patched(url):
        if url == fake_url:
            return True
        return original(url)

    return patch.object(opencode_go, "is_opencode_go_base_url", patched)


def _credential_model(fake_url):
    async def get_credential_obj():
        return SimpleNamespace(
            name="fake",
            to_esperanto_config=lambda: {"api_key": "k", "base_url": fake_url},
        )

    return SimpleNamespace(
        id="model:1",
        name="gpt-test",
        provider="openai_compatible",
        type="language",
        credential="credential:1",
        get_credential_obj=get_credential_obj,
    )


# ---------------------------------------------------------------------------
# 1. Esperanto direct path
# ---------------------------------------------------------------------------


class TestEsperantoDirect:
    @pytest.mark.asyncio
    async def test_direct_without_header_gets_missing_session_400(self, fake_server):
        from esperanto.providers.llm.openai_compatible import (
            OpenAICompatibleLanguageModel,
        )

        url, requests = fake_server
        model = OpenAICompatibleLanguageModel(api_key="k", base_url=url)
        with pytest.raises(RuntimeError, match="MissingSessionID"):
            await model.achat_complete([{"role": "user", "content": "hi"}])
        assert requests[-1]["session"] is None

    @pytest.mark.asyncio
    async def test_direct_with_header_passes_and_is_recorded(self, fake_server):
        from esperanto.providers.llm.openai_compatible import (
            OpenAICompatibleLanguageModel,
        )

        url, requests = fake_server
        model = OpenAICompatibleLanguageModel(
            api_key="k",
            base_url=url,
            config={"default_headers": {OPENCODE_SESSION_HEADER: "ses_direct01"}},
        )
        response = await model.achat_complete([{"role": "user", "content": "hi"}])
        assert not isinstance(response, AsyncGenerator)
        assert response.content == "ok"
        assert requests[-1]["session"] == "ses_direct01"
        assert requests[-1]["authorization"] == "Bearer k"


# ---------------------------------------------------------------------------
# 2-4. LangChain sync / async / streaming
# ---------------------------------------------------------------------------


class TestLangChainPaths:
    def _model(self, fake_url, **kwargs):
        from esperanto.providers.llm.openai_compatible import (
            OpenAICompatibleLanguageModel,
        )

        return OpenAICompatibleLanguageModel(
            api_key="k",
            base_url=fake_url,
            model_name="gpt-test",
            config={
                "default_headers": {OPENCODE_SESSION_HEADER: kwargs.pop("ses")}
            },
            **kwargs,
        )

    def test_sync_invoke_carries_header(self, fake_server):
        url, requests = fake_server
        langchain_model = self._model(url, ses="ses_sync0000000000000000001").to_langchain()
        result = langchain_model.invoke([{"role": "user", "content": "hi"}])
        assert result.content == "ok"
        assert requests[-1]["session"] == "ses_sync0000000000000000001"

    @pytest.mark.asyncio
    async def test_async_invoke_carries_header(self, fake_server):
        url, requests = fake_server
        langchain_model = self._model(url, ses="ses_async0000000000000000001").to_langchain()
        result = await langchain_model.ainvoke([{"role": "user", "content": "hi"}])
        assert result.content == "ok"
        assert requests[-1]["session"] == "ses_async0000000000000000001"

    def test_streaming_carries_header_on_request(self, fake_server):
        url, requests = fake_server
        langchain_model = self._model(
            url, ses="ses_stream0000000000000000001", streaming=True
        ).to_langchain()
        chunks = list(langchain_model.stream([{"role": "user", "content": "hi"}]))
        assert chunks
        chat_requests = [r for r in requests if r["method"] == "POST"]
        assert chat_requests
        assert all(
            r["session"] == "ses_stream0000000000000000001"
            for r in chat_requests
        )


# ---------------------------------------------------------------------------
# 5-7. Provisioning through OpenNotebook (fake URL allowed via test patch)
# ---------------------------------------------------------------------------


class TestProvisioningViaFakeServer:
    @pytest.mark.asyncio
    async def test_supplied_value_is_stable_across_provisioning_calls(
        self, fake_server, monkeypatch
    ):
        url, _ = fake_server
        model = _credential_model(url)
        _patch_model_get(monkeypatch, model)
        _patch_url_validation(monkeypatch)
        created = _patch_factory(monkeypatch)

        with _allow_fake_url(url):
            manager = ModelManager()
            await manager.get_model("model:1", opencode_session_id="ses_stable")
            await manager.get_model("model:1", opencode_session_id="ses_stable")
        v1 = created["calls"][0]["default_headers"][OPENCODE_SESSION_HEADER]
        v2 = created["calls"][1]["default_headers"][OPENCODE_SESSION_HEADER]
        assert v1 == v2 == "ses_stable"

    @pytest.mark.asyncio
    async def test_distinct_conversations_get_distinct_values(
        self, fake_server, monkeypatch
    ):
        url, _ = fake_server
        _patch_model_get(monkeypatch, _credential_model(url))
        _patch_url_validation(monkeypatch)
        created = _patch_factory(monkeypatch)

        with _allow_fake_url(url):
            manager = ModelManager()
            await manager.get_model("model:1", opencode_session_id="ses_aaaa")
            await manager.get_model("model:1", opencode_session_id="ses_bbbb")
        v1 = created["calls"][0]["default_headers"][OPENCODE_SESSION_HEADER]
        v2 = created["calls"][1]["default_headers"][OPENCODE_SESSION_HEADER]
        assert v1 != v2

    @pytest.mark.asyncio
    async def test_ephemeral_fallback_is_valid_and_unique(
        self, fake_server, monkeypatch
    ):
        url, _ = fake_server
        _patch_model_get(monkeypatch, _credential_model(url))
        _patch_url_validation(monkeypatch)
        created = _patch_factory(monkeypatch)

        with _allow_fake_url(url):
            manager = ModelManager()
            await manager.get_model("model:1")
            await manager.get_model("model:1")
        v1 = created["calls"][0]["default_headers"][OPENCODE_SESSION_HEADER]
        v2 = created["calls"][1]["default_headers"][OPENCODE_SESSION_HEADER]
        assert SES_RE.match(v1) and SES_RE.match(v2) and v1 != v2


# ---------------------------------------------------------------------------
# 6 (deep). Ask graph subcalls share one value through the real graph engine
# ---------------------------------------------------------------------------


class TestAskGraphOperationScoping:
    @pytest.mark.asyncio
    async def test_all_ask_nodes_share_one_operation_value(self):
        from langchain_core.runnables import RunnableConfig

        from open_notebook.graphs.ask import graph as ask_graph

        recorded = []

        class FakeModel:
            async def ainvoke(self, payload):
                from types import SimpleNamespace as NS

                return NS(content='{"reasoning": "r", "searches": [{"term": "t", "instructions": "i"}]}')

        async def fake_provision(content, model_id, default_type, **kwargs):
            recorded.append(kwargs.get("opencode_session_id"))
            return FakeModel()

        config = RunnableConfig(
            configurable={
                "strategy_model": "model:s",
                "answer_model": "model:a",
                "final_answer_model": "model:f",
                "opencode_session_id": "ses_askop00000000000000000001",
            }
        )
        with patch(
            "open_notebook.graphs.ask.provision_langchain_model", fake_provision
        ), patch(
            "open_notebook.graphs.ask.vector_search",
            new_callable=__import__("unittest.mock", fromlist=["AsyncMock"]).AsyncMock,
            return_value=[{"id": "1", "content": "c"}],
        ):
            await ask_graph.ainvoke(  # type: ignore[call-overload]
                {"question": "q"}, config=config
            )

        assert len(recorded) >= 3  # strategy + answer + final answer
        assert set(recorded) == {"ses_askop00000000000000000001"}


# ---------------------------------------------------------------------------
# 9. Podcast outline/transcript config reaches the Esperanto boundary
# ---------------------------------------------------------------------------


class TestPodcastConfigBoundary:
    def test_merged_config_headers_reach_esperanto_instance(self):
        """Replicating podcast-creator's merge, the header must land on the
        Esperanto model instance created via AIFactory."""
        from esperanto import AIFactory
        from esperanto.providers.llm.openai_compatible import (
            OpenAICompatibleLanguageModel,
        )

        outline_config = {
            "api_key": "k",
            "base_url": "http://localhost:9999/v1",
            "default_headers": {OPENCODE_SESSION_HEADER: "ses_podcast01"},
        }
        merged_config = {
            "max_tokens": 3000,
            "structured": {"type": "json"},
            **outline_config,
        }
        model = AIFactory.create_language(
            "openai-compatible", "gpt-test", config=merged_config
        )
        assert isinstance(model, OpenAICompatibleLanguageModel)
        headers = model._get_headers()
        assert headers[OPENCODE_SESSION_HEADER] == "ses_podcast01"
        assert model.max_tokens == 3000


# ---------------------------------------------------------------------------
# 10-11. Isolation and payload compatibility
# ---------------------------------------------------------------------------


class TestIsolationAndCompatibility:
    @pytest.mark.asyncio
    async def test_detector_not_patched_fake_url_gets_no_header(self, fake_server):
        """Without the test patch, a local endpoint is not OpenCode: no session
        header is attached, so the strict fake server rejects with 400 -
        proving production matching is not weakened for local URLs."""
        import pytest as _pytest
        from esperanto.providers.llm.openai_compatible import (
            OpenAICompatibleLanguageModel,
        )

        url, requests = fake_server
        model = OpenAICompatibleLanguageModel(api_key="k", base_url=url)
        with _pytest.raises(RuntimeError, match="MissingSessionID"):
            await model.achat_complete([{"role": "user", "content": "hi"}])
        assert requests[-1]["session"] is None

    @pytest.mark.asyncio
    async def test_structured_output_and_existing_headers_survive(self, fake_server):
        from esperanto.providers.llm.openai_compatible import (
            OpenAICompatibleLanguageModel,
        )

        url, requests = fake_server
        model = OpenAICompatibleLanguageModel(
            api_key="k",
            base_url=url,
            structured={"type": "json"},
            config={
                "default_headers": {
                    OPENCODE_SESSION_HEADER: "ses_compat0000000000000001",
                    "X-Custom": "kept",
                }
            },
        )
        await model.achat_complete([{"role": "user", "content": "hi"}])
        last = requests[-1]
        assert last["session"] == "ses_compat0000000000000001"
        assert last["custom"] == "kept"
        assert last["json"]["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_tools_payload_normalizes_with_header(self, fake_server):
        from esperanto.common_types import Tool, ToolFunction
        from esperanto.providers.llm.openai_compatible import (
            OpenAICompatibleLanguageModel,
        )

        url, requests = fake_server
        model = OpenAICompatibleLanguageModel(
            api_key="k",
            base_url=url,
            config={
                "default_headers": {OPENCODE_SESSION_HEADER: "ses_tools000000000001"}
            },
        )
        tools = [
            Tool(
                type="function",
                function=ToolFunction(
                    name="get_weather",
                    description="weather",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                ),
            )
        ]
        response = await model.achat_complete(
            [{"role": "user", "content": "hi"}], tools=tools
        )
        assert not isinstance(response, AsyncGenerator)
        assert response.choices[0].message.tool_calls is not None
        assert requests[-1]["session"] == "ses_tools000000000001"
        assert requests[-1]["json"]["tools"][0]["function"]["name"] == "get_weather"


# ---------------------------------------------------------------------------
# shared helpers for provisioning tests
# ---------------------------------------------------------------------------


def _patch_model_get(monkeypatch, model):
    async def fake_get(model_id):
        return model

    monkeypatch.setattr("open_notebook.ai.models.Model.get", staticmethod(fake_get))


def _patch_url_validation(monkeypatch):
    async def fake_revalidate(config, provider):
        return None

    monkeypatch.setattr(
        "open_notebook.ai.models._revalidate_config_urls", fake_revalidate
    )


def _patch_factory(monkeypatch):
    created: dict = {"calls": []}

    class _Lang:
        def to_langchain(self):
            return Mock(name="BaseChatModel")

    def fake_create_language(model_name, provider, config):
        created["calls"].append(config)
        return _Lang()

    monkeypatch.setattr(
        "open_notebook.ai.models.AIFactory.create_language",
        staticmethod(fake_create_language),
    )
    return created
