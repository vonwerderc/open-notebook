"""OpenCode Go session identity helpers.

A tiny, dependency-free seam for attaching a stable, opaque
``x-opencode-session`` header to inference requests aimed at the OpenCode Go
endpoint (``https://opencode.ai/zen/go/v1``) and nothing else. All other
providers are untouched.
"""

import uuid
from urllib.parse import urlsplit

OPENCODE_SESSION_HEADER = "x-opencode-session"

_OPENCODE_HOST = "opencode.ai"
_OPENCODE_PATH = "/zen/go/v1"


def is_opencode_go_base_url(base_url: str | None) -> bool:
    """Return True only for the exact OpenCode Go endpoint.

    Requires scheme ``https``, hostname ``opencode.ai`` (case-insensitive),
    port absent or ``443``, and normalized path exactly ``/zen/go/v1``.
    Query and fragment are ignored; one trailing slash is tolerated.
    Everything else - lookalike hosts, other ``/zen/*`` routes, plain HTTP -
    is rejected. Never raises.
    """
    if not base_url or not isinstance(base_url, str):
        return False
    try:
        parts = urlsplit(base_url.strip())
    except ValueError:
        return False
    if parts.scheme.lower() != "https":
        return False
    if parts.hostname is None or parts.hostname.lower() != _OPENCODE_HOST:
        return False
    if parts.port not in (None, 443):
        return False
    path = parts.path or ""
    if path.endswith("/") and path != "/":
        path = path[:-1]
    return path == _OPENCODE_PATH


def persistent_opencode_session_id(raw_identity: str) -> str:
    """Derive a stable, opaque session ID from an existing stable ID.

    The raw identity (e.g. a LangGraph ``thread_id``) never leaves the
    process: the header carries only the derived ``ses_`` value, which is
    deterministic, fixed-width, and reveals nothing about the source ID.
    """
    digest = uuid.uuid5(uuid.NAMESPACE_URL, f"open-notebook:{raw_identity}").hex
    return f"ses_{digest}"


def new_opencode_session_id() -> str:
    """Create one ephemeral session ID for a single stateless operation."""
    return f"ses_{uuid.uuid4().hex}"


def headers_for_opencode_go(
    base_url: str | None,
    session_id: str,
    existing: dict[str, str] | None = None,
) -> dict[str, str] | None:
    """Build request headers for ``base_url``, or None for non-OpenCode URLs.

    Returns a new mapping that preserves any unrelated existing headers and
    sets ``x-opencode-session`` to ``session_id``. The caller's mapping is
    never mutated.
    """
    if not is_opencode_go_base_url(base_url):
        return None
    headers = dict(existing or {})
    headers[OPENCODE_SESSION_HEADER] = session_id
    return headers
