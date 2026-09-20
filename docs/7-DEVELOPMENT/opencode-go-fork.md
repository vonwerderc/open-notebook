# OpenCode Go Compatibility Fork

Why this fork exists, what it changes, and how to maintain it.

## Why

OpenCode Go (`https://opencode.ai/zen/go/v1`) requires an `x-opencode-session`
header on every inference request and returns HTTP 400 `MissingSessionID`
otherwise. Upstream OpenNotebook/Esperanto (at the pinned versions below) have
no way to attach a per-conversation header, so every request fails. This fork
attaches a stable, opaque session header **only** for that endpoint; all other
providers are byte-for-byte unchanged.

## Commits

| Component | Baseline | Fork ref |
|---|---|---|
| OpenNotebook | `lfnovo/open-notebook` `v1.14.0` (`30c7e2a63e43b7f270fc2c638f0b6246934a53f4`) | `vonwerderc/open-notebook` `main` (from `be753989`) |
| Esperanto | `lfnovo/esperanto` `v2.25.1` (`dbded66df5f1f23f3a5254cbc9c25d58cc88f9ab`) | `vonwerderc/esperanto` `c7ee427a82c480495095cf83c47101f611354712` |

The Esperanto fork is one commit: optional `default_headers` on
`OpenAICompatibleLanguageModel` (direct HTTP merge under provider-controlled
headers, plus LangChain `ChatOpenAI` forwarding). OpenNotebook pins it by
immutable SHA in `pyproject.toml` / `uv.lock`.

## Session identity rules

- Endpoint match: scheme `https`, host `opencode.ai` (case-insensitive), port
  absent or 443, path exactly `/zen/go/v1` (one trailing slash tolerated).
  Everything else - lookalike hosts, other `/zen/*` routes - is rejected.
- Header name: `x-opencode-session`; values are opaque `ses_` + 32 hex chars.
- Persistent (notebook chat, source chat): `uuid5(NAMESPACE_URL,
  "open-notebook:" + thread_id)` - stable across turns, distinct per session.
- Ephemeral (Ask, transformations, note titles, source transforms, model test,
  credential test, discovery): one `uuid4` per operation, shared by subcalls.
- Podcasts: one value per generation job derived from the command ID, shared by
  outline/transcript configs; never applied to TTS.
- Raw record IDs are never sent or logged.

## Tests

```bash
uv run pytest tests/ -q                      # full suite (714 passing at fork main)
uv run pytest tests/test_opencode_go.py tests/test_opencode_session_headers_integration.py -v
uv run ruff check . && uv run python -m mypy .
```

## Real smoke (no secrets in shell)

The credential lives in the Hermes env store (`OPENCODE_GO_API_KEY`). Smoke
procedure: disposable SurrealDB + app containers on a scratch network; create
`openai_compatible` credential at `https://opencode.ai/zen/go/v1` via
`POST /api/credentials`; run `/test`, `/discover`, register a language model,
`POST /api/models/{id}/test`, two `/api/chat/execute` turns, one
`/api/transformations/execute`; then verify zero `MissingSessionID` in logs and
remove the stack. Generated `ses_*` values must never appear in logs.

## Images

| Tag | Purpose |
|---|---|
| `ghcr.io/vonwerderc/open-notebook:v1.14.0-opencodego.1` | pinned release of this fork |
| `ghcr.io/vonwerderc/open-notebook:v1-opencodego` | moving channel, same digest |
| `lfnovo/open_notebook:v1-latest` | official rollback target |

Both GHCR tags point at digest
`sha256:f79938da435553aeee665873b5226cd8e0c4d82034325cbcefe8fcf6738f2e37`.

If Dockhand is ever pointed at this fork, use `v1-opencodego`; back up the live
deployment source of truth and record the currently-running digest first.

## Update procedure

1. Fetch the upstream release; create a new branch from this fork's `main`.
2. Rebase/re-pin and re-run the full test matrix above.
3. Check whether upstream OpenNotebook/Esperanto now support generic default
   headers or native OpenCode session affinity - if so, drop this fork.
