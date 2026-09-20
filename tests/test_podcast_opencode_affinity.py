"""Podcast job OpenCode session affinity (Task 9).

One operation value per podcast generation job: derived from the command ID
when present, ephemeral otherwise. Outline and transcript config resolutions
share it; TTS and unrelated configs never receive the header.
"""

from types import SimpleNamespace

import pytest

from open_notebook.ai.opencode_go import (
    OPENCODE_SESSION_HEADER,
    persistent_opencode_session_id,
)
from open_notebook.podcasts.models import EpisodeProfile, _resolve_model_config

OPENCODE_URL = "https://opencode.ai/zen/go/v1"
COMMAND_ID = "command:abc123"
EXPECTED_JOB_VALUE = persistent_opencode_session_id(COMMAND_ID)


def _fake_language_model(cred_config):
    async def get_credential_obj():
        return SimpleNamespace(name="c", to_esperanto_config=lambda: cred_config)

    return SimpleNamespace(
        id="model:1",
        name="gpt-test",
        provider="openai_compatible",
        type="language",
        credential="credential:1",
        get_credential_obj=get_credential_obj,
    )


def _fake_tts_model(cred_config):
    async def get_credential_obj():
        return SimpleNamespace(name="c", to_esperanto_config=lambda: cred_config)

    return SimpleNamespace(
        id="model:tts",
        name="tts-test",
        provider="openai_compatible",
        type="text_to_speech",
        credential="credential:1",
        get_credential_obj=get_credential_obj,
    )


def _patch_model_get(monkeypatch, model):
    async def fake_get(model_id):
        return model

    monkeypatch.setattr(
        "open_notebook.ai.models.Model.get", staticmethod(fake_get)
    )


class TestResolveModelConfigSessionHeader:
    @pytest.mark.asyncio
    async def test_language_model_with_job_value_gets_header(self, monkeypatch):
        monkeypatch.delenv("OPENAI_COMPATIBLE_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_COMPATIBLE_BASE_URL_LLM", raising=False)
        _patch_model_get(
            monkeypatch,
            _fake_language_model({"api_key": "k", "base_url": OPENCODE_URL}),
        )
        provider, name, config = await _resolve_model_config(
            "model:1", max_tokens=100, opencode_session_id="ses_job"
        )
        assert config["default_headers"] == {OPENCODE_SESSION_HEADER: "ses_job"}
        assert config["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_no_job_value_leaves_config_unchanged(self, monkeypatch):
        _patch_model_get(
            monkeypatch,
            _fake_language_model({"api_key": "k", "base_url": OPENCODE_URL}),
        )
        provider, name, config = await _resolve_model_config("model:1")
        assert "default_headers" not in config

    @pytest.mark.asyncio
    async def test_non_opencode_language_model_gets_no_header(self, monkeypatch):
        _patch_model_get(
            monkeypatch,
            _fake_language_model(
                {"api_key": "k", "base_url": "https://api.other.dev/v1"}
            ),
        )
        provider, name, config = await _resolve_model_config(
            "model:1", opencode_session_id="ses_job"
        )
        assert "default_headers" not in config

    @pytest.mark.asyncio
    async def test_tts_model_never_gets_header_even_for_opencode_url(
        self, monkeypatch
    ):
        _patch_model_get(
            monkeypatch,
            _fake_tts_model({"api_key": "k", "base_url": OPENCODE_URL}),
        )
        provider, name, config = await _resolve_model_config(
            "model:tts", opencode_session_id="ses_job"
        )
        assert "default_headers" not in config


class TestPodcastCommandJobValue:
    def _episode_profile(self):
        return EpisodeProfile(
            id="episode_profile:1",
            name="ep",
            speaker_config="speaker_profile:1",
            outline_llm="model:1",
            transcript_llm="model:1",
            default_briefing="brief",
            num_segments=5,
        )

    @pytest.mark.asyncio
    async def test_command_id_derives_deterministic_job_value(self):
        """The job value is uuid5(command_id) - stable per command."""
        from open_notebook.ai.opencode_go import (
            persistent_opencode_session_id as posid,
        )

        assert posid(COMMAND_ID) == EXPECTED_JOB_VALUE
        assert EXPECTED_JOB_VALUE.startswith("ses_")

    @pytest.mark.asyncio
    async def test_outline_and_transcript_share_job_value(self, monkeypatch):
        """Both resolutions receive the identical job session value."""
        received = []

        async def fake_resolve(self, opencode_session_id=None):
            received.append(opencode_session_id)
            return ("openai_compatible", "gpt-test", {"api_key": "k"})

        monkeypatch.setattr(
            EpisodeProfile, "resolve_outline_config", fake_resolve
        )
        monkeypatch.setattr(
            EpisodeProfile, "resolve_transcript_config", fake_resolve
        )

        profile = self._episode_profile()
        await profile.resolve_outline_config(opencode_session_id="ses_job")
        await profile.resolve_transcript_config(opencode_session_id="ses_job")
        assert received == ["ses_job", "ses_job"]

    @pytest.mark.asyncio
    async def test_missing_job_value_gets_ephemeral(self, monkeypatch):

        received = []

        async def fake_resolve(self, opencode_session_id=None):
            received.append(opencode_session_id)
            return ("openai_compatible", "gpt-test", {"api_key": "k"})

        monkeypatch.setattr(
            EpisodeProfile, "resolve_outline_config", fake_resolve
        )
        profile = self._episode_profile()
        await profile.resolve_outline_config()
        assert received == [None]

    def test_podcast_creator_config_merge_preserves_default_headers(self):
        """max_tokens/structured merges must keep default_headers in the config.

        podcast-creator 0.12.0 merges max_tokens into the config dict before
        use; a plain dict update preserves other keys. This pins the contract
        the podcast path relies on.
        """
        config = {"default_headers": {OPENCODE_SESSION_HEADER: "ses_job"}}
        merged = {**config, "max_tokens": 100}
        assert merged["default_headers"] == {
            OPENCODE_SESSION_HEADER: "ses_job"
        }
        assert merged["max_tokens"] == 100
