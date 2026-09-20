"""OpenCode Go session identity helpers: detector, IDs, header merging."""

import re
import uuid

import pytest

from open_notebook.ai.opencode_go import (
    OPENCODE_SESSION_HEADER,
    headers_for_opencode_go,
    is_opencode_go_base_url,
    new_opencode_session_id,
    persistent_opencode_session_id,
)

SES_RE = re.compile(r"^ses_[0-9a-f]{32}$")


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
        assert headers[OPENCODE_SESSION_HEADER] == "ses_abc"
        assert headers["Authorization"] == "Bearer k"
        assert headers["X-Custom"] == "1"

    def test_does_not_mutate_caller_mapping(self):
        existing = {"Authorization": "Bearer k"}
        headers_for_opencode_go("https://opencode.ai/zen/go/v1", "ses_abc", existing)
        assert existing == {"Authorization": "Bearer k"}

    def test_header_name_literal(self):
        assert OPENCODE_SESSION_HEADER == "x-opencode-session"
