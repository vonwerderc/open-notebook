"""OpenCode Go session identity helpers: detector, IDs, header merging."""

import re
import uuid
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from open_notebook.ai.opencode_go import (
    OPENCODE_SESSION_HEADER,
    headers_for_opencode_go,
    is_opencode_go_base_url,
    new_opencode_session_id,
    persistent_opencode_session_id,
)

SES_RE = re.compile(r"^ses_[0-9a-f]{32}$")

OPENCODE_URL = "https://opencode.ai/zen/go/v1"


class TestDetector:
    """Only the exact https opencode.ai /zen/go/v1 endpoint matches."""

    @pytest.mark.parametrize(
        "url",
        [
            "https://opencode.ai/zen/go/v1",
            "https://OPENCODE.AI/zen/go/v1/",
            "https://opencode.ai:443/zen/go/v1?x=1",
        ],
    )
    def test_matches(self, url):
        assert is_opencode_go_base_url(url) is True

    @pytest.mark.parametrize(
        "url",
        [
            "http://opencode.ai/zen/go/v1",
            "https://opencode.ai:8443/zen/go/v1",
            "https://opencode.ai/zen/go",
            "https://opencode.ai/zen/free/v1",
            "https://opencode.ai.evil.example/zen/go/v1",
            "https://proxy.example/opencode.ai/zen/go/v1",
            "https://opencode.ai/zen/go/v1/extra",
            "https://badhost/zen/go/v1",
        ],
    )
    def test_rejects(self, url):
        assert is_opencode_go_base_url(url) is False

    @pytest.mark.parametrize("url", [None, "", "   ", "https://", "://bad", "https://:443/zen/go/v1"])
    def test_malformed_rejected_without_raising(self, url):
        assert is_opencode_go_base_url(url) is False


class TestSessionIds:
    """Persistent IDs are deterministic; ephemeral IDs are unique; both shaped."""

    def test_persistent_deterministic_and_shaped(self):
        v1 = persistent_opencode_session_id("thread-abc")
        v2 = persistent_opencode_session_id("thread-abc")
        assert v1 == v2
        assert SES_RE.match(v1)

    def test_persistent_distinct_identities_differ(self):
        assert persistent_opencode_session_id("a") != persistent_opencode_session_id("b")

    def test_ephemeral_unique_and_shaped(self):
        a = new_opencode_session_id()
        b = new_opencode_session_id()
        assert a != b
        assert SES_RE.match(a)
        assert SES_RE.match(b)

    def test_persistent_matches_uuid5_contract(self):
        expected = "ses_" + uuid.uuid5(
            uuid.NAMESPACE_URL, "open-notebook:thread-abc"
        ).hex
        assert persistent_opencode_session_id("thread-abc") == expected


class TestHeaderMerging:
    """Headers are provider-isolated and merged without mutating input."""

    def test_returns_none_for_non_opencode(self):
        assert headers_for_opencode_go("https://openrouter.ai/api/v1", "ses_x") is None
        assert headers_for_opencode_go(None, "ses_x") is None

    def test_sets_header_on_fresh_mapping(self):
        headers = headers_for_opencode_go("https://opencode.ai/zen/go/v1", "ses_abc")
        assert headers == {OPENCODE_SESSION_HEADER: "ses_abc"}

    def test_preserves_existing_headers(self):
        existing = {"Authorization": "Bearer k", "X-Custom": "1"}
        headers = headers_for_opencode_go(
            "https://opencode.ai/zen/go/v1", "ses_abc", existing
        )
        assert headers is not None
        assert headers[OPENCODE_SESSION_HEADER] == "ses_abc"
        assert headers["Authorization"] == "Bearer k"
        assert headers["X-Custom"] == "1"

    def test_does_not_mutate_caller_mapping(self):
        existing = {"Authorization": "Bearer k"}
        headers_for_opencode_go("https://opencode.ai/zen/go/v1", "ses_abc", existing)
        assert existing == {"Authorization": "Bearer k"}

    def test_header_name_literal(self):
        assert OPENCODE_SESSION_HEADER == "x-opencode-session"


# ---------------------------------------------------------------------------
# Task 5: header injection at the resolved model boundary
# ---------------------------------------------------------------------------

from open_notebook.ai.models import ModelManager
from open_notebook.ai.provision import provision_langchain_model


def _fake_model(provider="openai_compatible", model_type="language", credential=None, cred_config=None):
    async def get_credential_obj():
        if cred_config is None:
            return None
        return SimpleNamespace(name="cred", to_esperanto_config=lambda: cred_config)

    return SimpleNamespace(
        id="model:1",
        name="gpt-test",
        provider=provider,
        type=model_type,
        credential=credential,
        get_credential_obj=get_credential_obj,
    )


def _patch_model_loader(monkeypatch, model):
    async def fake_get(model_id):
        return model

    monkeypatch.setattr(
        "open_notebook.ai.models.Model.get", staticmethod(fake_get)
    )


def _patch_url_validation(monkeypatch):
    async def fake_revalidate(config, provider):
        return None

    monkeypatch.setattr(
        "open_notebook.ai.models._revalidate_config_urls", fake_revalidate
    )


def _patch_provider_keys(monkeypatch):
    async def fake_keys(provider):
        return None

    monkeypatch.setattr(
        "open_notebook.ai.key_provider.provision_provider_keys", fake_keys
    )


def _patch_factory(monkeypatch):
    from esperanto import LanguageModel

    created = {}

    class _FakeLanguageModel(LanguageModel):
        @property
        def provider(self):
            return "openai-compatible"

        def chat_complete(self, messages, **kwargs):  # type: ignore[override]
            raise NotImplementedError

        async def achat_complete(self, messages, **kwargs):  # type: ignore[override]
            raise NotImplementedError

        def _get_models(self):
            return []

        def _get_default_model(self):
            return ""

        def to_langchain(self):
            return Mock(name="BaseChatModel")

    def fake_create_language(model_name, provider, config):
        created["model_name"] = model_name
        created["provider"] = provider
        created["config"] = config
        return _FakeLanguageModel()

    monkeypatch.setattr(
        "open_notebook.ai.models.AIFactory.create_language",
        staticmethod(fake_create_language),
    )
    return created


OPENCODE_CFG = {"api_key": "k", "base_url": OPENCODE_URL}


class TestModelManagerSessionHeader:
    """ModelManager.get_model() attaches the header only for OpenCode Go."""

    @pytest.mark.asyncio
    async def test_credential_backed_opencode_url_receives_supplied_header(self, monkeypatch):
        model = _fake_model(credential="credential:1", cred_config=dict(OPENCODE_CFG))
        _patch_model_loader(monkeypatch, model)
        _patch_url_validation(monkeypatch)
        created = _patch_factory(monkeypatch)

        manager = ModelManager()
        result = await manager.get_model(
            "model:1", opencode_session_id="ses_persistent"
        )
        assert result is not None
        assert created["config"]["default_headers"] == {
            OPENCODE_SESSION_HEADER: "ses_persistent"
        }
        assert "opencode_session_id" not in created["config"]

    @pytest.mark.asyncio
    async def test_env_backed_opencode_url_receives_header(self, monkeypatch):
        model = _fake_model()  # no credential -> env fallback
        _patch_model_loader(monkeypatch, model)
        _patch_provider_keys(monkeypatch)
        created = _patch_factory(monkeypatch)

        monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL_LLM", OPENCODE_URL)
        monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "envkey")

        manager = ModelManager()
        result = await manager.get_model(
            "model:1", opencode_session_id="ses_env"
        )
        assert result is not None
        assert created["config"]["default_headers"] == {
            OPENCODE_SESSION_HEADER: "ses_env"
        }

    @pytest.mark.asyncio
    async def test_missing_operation_id_receives_ephemeral_fallback(self, monkeypatch):
        model = _fake_model(credential="credential:1", cred_config=dict(OPENCODE_CFG))
        _patch_model_loader(monkeypatch, model)
        _patch_url_validation(monkeypatch)
        created = _patch_factory(monkeypatch)

        manager = ModelManager()
        result = await manager.get_model("model:1")
        assert result is not None
        value = created["config"]["default_headers"][OPENCODE_SESSION_HEADER]
        assert SES_RE.match(value)

    @pytest.mark.parametrize(
        "provider,base_url",
        [
            ("openrouter", "https://openrouter.ai/api/v1"),
            ("z_ai", "https://api.z.ai/api/paas/v4"),
            ("openai", None),
            ("openai_compatible", "https://localhost:8080/v1"),
            ("openai_compatible", "https://opencode.ai/zen/free/v1"),
            ("openai_compatible", "https://opencode.ai.evil.example/zen/go/v1"),
        ],
    )
    @pytest.mark.asyncio
    async def test_non_opencode_providers_receive_no_header(
        self, monkeypatch, provider, base_url
    ):
        cfg = {"api_key": "k"}
        if base_url:
            cfg["base_url"] = base_url
        model = _fake_model(provider=provider, credential="credential:1", cred_config=cfg)
        _patch_model_loader(monkeypatch, model)
        _patch_url_validation(monkeypatch)
        created = _patch_factory(monkeypatch)
        monkeypatch.delenv("OPENAI_COMPATIBLE_BASE_URL_LLM", raising=False)
        monkeypatch.delenv("OPENAI_COMPATIBLE_BASE_URL", raising=False)

        manager = ModelManager()
        result = await manager.get_model(
            "model:1", opencode_session_id="ses_should_not_leak"
        )
        assert result is not None
        headers = created["config"].get("default_headers")
        assert not headers or OPENCODE_SESSION_HEADER not in headers

    @pytest.mark.asyncio
    async def test_existing_custom_headers_retained(self, monkeypatch):
        cfg = {
            "api_key": "k",
            "base_url": OPENCODE_URL,
            "default_headers": {"X-Existing": "yes"},
        }
        model = _fake_model(credential="credential:1", cred_config=cfg)
        _patch_model_loader(monkeypatch, model)
        _patch_url_validation(monkeypatch)
        created = _patch_factory(monkeypatch)

        manager = ModelManager()
        result = await manager.get_model(
            "model:1", opencode_session_id="ses_keep"
        )
        assert result is not None
        headers = created["config"]["default_headers"]
        assert headers["X-Existing"] == "yes"
        assert headers[OPENCODE_SESSION_HEADER] == "ses_keep"

    @pytest.mark.asyncio
    async def test_large_context_fallback_keeps_same_session_id(self, monkeypatch):
        model = _fake_model(credential="credential:1", cred_config=dict(OPENCODE_CFG))
        _patch_model_loader(monkeypatch, model)
        _patch_url_validation(monkeypatch)
        created = _patch_factory(monkeypatch)

        async def fake_defaults():
            return SimpleNamespace(
                default_chat_model="model:1",
                default_transformation_model=None,
                large_context_model="model:1",
                default_text_to_speech_model=None,
                default_speech_to_text_model=None,
                default_embedding_model=None,
                default_tools_model=None,
            )

        monkeypatch.setattr(
            "open_notebook.ai.models.DefaultModels.get_instance",
            staticmethod(fake_defaults),
        )

        long_content = "x" * 200_000
        result = await provision_langchain_model(
            long_content,
            None,
            "chat",
            opencode_session_id="ses_large",
        )
        assert result is not None
        assert created["config"]["default_headers"] == {
            OPENCODE_SESSION_HEADER: "ses_large"
        }

    @pytest.mark.asyncio
    async def test_session_id_never_leaks_as_unknown_config_key(self, monkeypatch):
        model = _fake_model(credential="credential:1", cred_config=dict(OPENCODE_CFG))
        _patch_model_loader(monkeypatch, model)
        _patch_url_validation(monkeypatch)
        created = _patch_factory(monkeypatch)

        manager = ModelManager()
        await manager.get_model("model:1", opencode_session_id="ses_x")
        assert "opencode_session_id" not in created["config"]
        assert created["config"]["default_headers"] == {
            OPENCODE_SESSION_HEADER: "ses_x"
        }

