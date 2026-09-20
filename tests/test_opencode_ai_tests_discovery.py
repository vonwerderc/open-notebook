"""OpenCode headers in model tests, credential tests, and discovery (Task 8).

Direct Esperanto model tests, raw ``GET /models`` credential tests, and
discovery requests must carry ``x-opencode-session`` for the OpenCode Go
endpoint only; Authorization and DNS-pinning behavior stay intact.
"""

from types import SimpleNamespace

import httpx
import pytest

from open_notebook.ai import connection_tester, model_discovery
from open_notebook.ai.opencode_go import (
    OPENCODE_SESSION_HEADER,
)
from open_notebook.utils.url_validation import PinnedHttpTarget

OPENCODE_URL = "https://opencode.ai/zen/go/v1"
OTHER_URL = "https://api.example.com/v1"


def _fake_client_factory(requests, payload=None):
    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url, headers=None, extensions=None, timeout=None):
            requests.append(
                {"url": url, "headers": dict(headers or {}), "extensions": extensions}
            )
            return httpx.Response(
                200,
                json=payload or {"data": [{"id": "test-model"}]},
                request=httpx.Request("GET", url),
            )

    return FakeAsyncClient


def _fake_prepare_pinned(monkeypatch, module):
    async def fake_prepare(url, provider):
        return PinnedHttpTarget(url=url, headers={}, extensions={})

    monkeypatch.setattr(module, "prepare_pinned_http_target", fake_prepare)


class TestCredentialTestOpenAICompatible:
    @pytest.mark.asyncio
    async def test_opencode_url_sends_session_header(self, monkeypatch):
        requests: list[dict] = []
        _fake_prepare_pinned(monkeypatch, connection_tester)
        monkeypatch.setattr(
            connection_tester.httpx,
            "AsyncClient",
            _fake_client_factory(requests),
        )

        ok, msg = await connection_tester._test_openai_compatible_connection(
            OPENCODE_URL, "sk-test"
        )
        assert ok is True
        assert len(requests) == 1
        assert OPENCODE_SESSION_HEADER in requests[0]["headers"]
        assert requests[0]["headers"][OPENCODE_SESSION_HEADER].startswith("ses_")
        assert requests[0]["headers"]["Authorization"] == "Bearer sk-test"

    @pytest.mark.asyncio
    async def test_non_opencode_url_sends_no_session_header(self, monkeypatch):
        requests: list[dict] = []
        _fake_prepare_pinned(monkeypatch, connection_tester)
        monkeypatch.setattr(
            connection_tester.httpx,
            "AsyncClient",
            _fake_client_factory(requests),
        )

        ok, msg = await connection_tester._test_openai_compatible_connection(
            OTHER_URL, "sk-test"
        )
        assert ok is True
        assert len(requests) == 1
        assert OPENCODE_SESSION_HEADER not in requests[0]["headers"]


class TestDiscoveryOpenAICompatible:
    @pytest.mark.asyncio
    async def test_opencode_discovery_sends_session_header(self, monkeypatch):
        requests: list[dict] = []
        _fake_prepare_pinned(monkeypatch, model_discovery)
        monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", OPENCODE_URL)
        monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "sk-test")
        monkeypatch.setattr(
            model_discovery.httpx,
            "AsyncClient",
            _fake_client_factory(requests),
        )

        models = await model_discovery.discover_openai_compatible_models()
        assert [m.name for m in models] == ["test-model"]
        assert len(requests) == 1
        assert OPENCODE_SESSION_HEADER in requests[0]["headers"]
        assert requests[0]["headers"][OPENCODE_SESSION_HEADER].startswith("ses_")
        assert requests[0]["headers"]["Authorization"] == "Bearer sk-test"

    @pytest.mark.asyncio
    async def test_non_opencode_discovery_sends_no_session_header(self, monkeypatch):
        requests: list[dict] = []
        _fake_prepare_pinned(monkeypatch, model_discovery)
        monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", OTHER_URL)
        monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "sk-test")
        monkeypatch.setattr(
            model_discovery.httpx,
            "AsyncClient",
            _fake_client_factory(requests),
        )

        models = await model_discovery.discover_openai_compatible_models()
        assert [m.name for m in models] == ["test-model"]
        assert OPENCODE_SESSION_HEADER not in requests[0]["headers"]


class TestIndividualModelTest:
    @pytest.mark.asyncio
    async def test_language_model_test_passes_operation_id_to_provisioning(self):
        from unittest.mock import patch

        from esperanto import LanguageModel

        captured = {}

        class FakeLanguageModel(LanguageModel):
            @property
            def provider(self):
                return "openai-compatible"

            async def achat_complete(self, messages, **kwargs):  # type: ignore[override]
                captured["called"] = True
                return SimpleNamespace(content="hello")  # type: ignore[return-value]

            def chat_complete(self, messages, **kwargs):  # type: ignore[override]
                raise NotImplementedError

            def _get_models(self):
                return []

            def _get_default_model(self):
                return ""

            def to_langchain(self):
                raise NotImplementedError

        async def fake_get_model(self, model_id, opencode_session_id=None, **kwargs):
            captured["opencode_session_id"] = opencode_session_id
            return FakeLanguageModel()

        import open_notebook.ai.models as models_module

        model = SimpleNamespace(
            id="model:1", type="language", provider="openai_compatible"
        )
        with patch.object(models_module.ModelManager, "get_model", fake_get_model):
            ok, msg = await connection_tester.test_individual_model(model)

        assert captured["called"] is True
        value = str(captured["opencode_session_id"])
        assert value.startswith("ses_")
