"""OpenCode session-affinity wiring for notebook and source chats (Task 6).

The persistent session value must derive deterministically from the full
thread_id at each router invocation, travel through RunnableConfig, and reach
``provision_langchain_model`` explicitly in each graph node.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from open_notebook.ai.opencode_go import persistent_opencode_session_id

NB_SESSION = "chat_session:nb1"
SRC_SESSION = "chat_session:src1"
NB_VALUE = persistent_opencode_session_id(NB_SESSION)
SRC_VALUE = persistent_opencode_session_id(SRC_SESSION)


@pytest.fixture
def client():
    from api.main import app

    return TestClient(app)


def _session(**overrides):
    defaults = dict(
        id=NB_SESSION,
        title="S",
        created="2026-01-01",
        updated="2026-01-02",
        model_override=None,
        save=AsyncMock(),
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _chat_result():
    msg = SimpleNamespace(id="m1", type="ai", content="ok")
    return {"messages": [msg]}


def _captured_config(mock_graph):
    _, kwargs = mock_graph.invoke.call_args
    return kwargs["config"]["configurable"]


class TestNotebookChatRouter:
    @pytest.mark.asyncio
    @patch("api.routers.chat.repo_query", new_callable=AsyncMock)
    @patch("api.routers.chat.Notebook.get", new_callable=AsyncMock)
    @patch("api.routers.chat.get_session_or_404", new_callable=AsyncMock)
    @patch("api.routers.chat.chat_graph")
    async def test_execute_chat_configures_persistent_session_id(
        self, mock_graph, mock_get_session, mock_nb_get, mock_repo, client
    ):
        mock_get_session.return_value = (NB_SESSION, _session())
        mock_nb_get.return_value = SimpleNamespace(id="notebook:1")
        mock_repo.return_value = [{"out": "notebook:1"}]
        mock_graph.get_state.return_value = MagicMock(values={"messages": []})
        mock_graph.invoke.return_value = _chat_result()

        resp = client.post(
            "/api/chat/execute",
            json={
                "session_id": "nb1",
                "message": "hi",
                "context": {},
            },
        )

        assert resp.status_code == 200
        cfg = _captured_config(mock_graph)
        assert cfg["opencode_session_id"] == NB_VALUE
        assert cfg["thread_id"] == NB_SESSION

    @pytest.mark.asyncio
    @patch("api.routers.chat.repo_query", new_callable=AsyncMock)
    @patch("api.routers.chat.Notebook.get", new_callable=AsyncMock)
    @patch("api.routers.chat.get_session_or_404", new_callable=AsyncMock)
    @patch("api.routers.chat.chat_graph")
    async def test_two_turns_share_one_value(
        self, mock_graph, mock_get_session, mock_nb_get, mock_repo, client
    ):
        mock_get_session.return_value = (NB_SESSION, _session())
        mock_nb_get.return_value = SimpleNamespace(id="notebook:1")
        mock_repo.return_value = [{"out": "notebook:1"}]
        mock_graph.get_state.return_value = MagicMock(values={"messages": []})
        mock_graph.invoke.return_value = _chat_result()

        for _ in range(2):
            resp = client.post(
                "/api/chat/execute",
                json={"session_id": "nb1", "message": "hi", "context": {}},
            )
            assert resp.status_code == 200

        assert mock_graph.invoke.call_count == 2
        first = mock_graph.invoke.call_args_list[0]
        second = mock_graph.invoke.call_args_list[1]
        v1 = first.kwargs["config"]["configurable"]["opencode_session_id"]
        v2 = second.kwargs["config"]["configurable"]["opencode_session_id"]
        assert v1 == v2 == NB_VALUE

    @pytest.mark.asyncio
    @patch("api.routers.chat.repo_query", new_callable=AsyncMock)
    @patch("api.routers.chat.Notebook.get", new_callable=AsyncMock)
    @patch("api.routers.chat.get_session_or_404", new_callable=AsyncMock)
    @patch("api.routers.chat.chat_graph")
    async def test_model_override_does_not_change_session_value(
        self, mock_graph, mock_get_session, mock_nb_get, mock_repo, client
    ):
        mock_get_session.return_value = (
            NB_SESSION,
            _session(model_override="model:session_default"),
        )
        mock_nb_get.return_value = SimpleNamespace(id="notebook:1")
        mock_repo.return_value = [{"out": "notebook:1"}]
        mock_graph.get_state.return_value = MagicMock(values={"messages": []})
        mock_graph.invoke.return_value = _chat_result()

        resp = client.post(
            "/api/chat/execute",
            json={
                "session_id": "nb1",
                "message": "hi",
                "context": {},
                "model_override": "model:other",
            },
        )
        assert resp.status_code == 200
        cfg = _captured_config(mock_graph)
        assert cfg["opencode_session_id"] == NB_VALUE
        assert cfg["model_id"] == "model:other"


class TestSourceChatRouter:
    @pytest.mark.asyncio
    @patch("api.routers.source_chat.get_verified_source_session", new_callable=AsyncMock)
    @patch("api.routers.source_chat.source_chat_graph")
    async def test_source_chat_uses_independent_session_value(
        self, mock_graph, mock_verified, client
    ):
        mock_verified.return_value = (
            "source:xyz",
            SimpleNamespace(id="source:xyz"),
            SRC_SESSION,
            _session(),
        )
        mock_graph.get_state.return_value = MagicMock(values={"messages": []})
        mock_graph.invoke.return_value = {
            "messages": [SimpleNamespace(id="m1", type="ai", content="ok")]
        }

        resp = client.post(
            "/api/sources/xyz/chat/sessions/src1/messages",
            json={"message": "hi"},
        )
        assert resp.status_code == 200
        # Consume the SSE stream so the generator body actually runs.
        body = resp.text
        assert "ai_message" in body
        cfg = _captured_config(mock_graph)
        assert cfg["opencode_session_id"] == SRC_VALUE
        assert cfg["opencode_session_id"] != NB_VALUE


class TestGraphNodesPassValueToProvisioning:
    @pytest.mark.asyncio
    async def test_chat_node_passes_session_id_to_provision(self):
        from langchain_core.messages import HumanMessage

        from open_notebook.graphs.chat import call_model_with_messages

        with patch(
            "open_notebook.graphs.chat.provision_langchain_model",
            new_callable=AsyncMock,
        ) as mock_provision, patch("open_notebook.graphs.chat.Prompter") as mock_prompter:
            mock_prompter.return_value.render.return_value = "SYS"
            mock_model = MagicMock()
            mock_model.invoke.return_value = SimpleNamespace(
                content="ok",
                model_copy=lambda update=None: SimpleNamespace(
                    content="ok", id="m", type="ai"
                ),
            )
            mock_provision.return_value = mock_model

            state = {"messages": [HumanMessage(content="hi")]}
            config = {
                "configurable": {
                    "thread_id": NB_SESSION,
                    "opencode_session_id": NB_VALUE,
                }
            }
            call_model_with_messages(state, config)

        assert mock_provision.await_count == 1
        assert (
            mock_provision.await_args.kwargs["opencode_session_id"] == NB_VALUE
        )

    @pytest.mark.asyncio
    async def test_source_chat_node_passes_session_id_to_provision(self):
        from langchain_core.messages import HumanMessage

        from open_notebook.graphs.source_chat import (
            _call_model_with_source_context_inner,
        )

        with patch(
            "open_notebook.graphs.source_chat.provision_langchain_model",
            new_callable=AsyncMock,
        ) as mock_provision, patch(
            "open_notebook.graphs.source_chat.build_source_context",
            new_callable=AsyncMock,
        ) as mock_ctx, patch(
            "open_notebook.graphs.source_chat.Prompter"
        ) as mock_prompter:
            mock_ctx.return_value = {"sources": [], "insights": []}
            mock_prompter.return_value.render.return_value = "SYS"
            mock_model = MagicMock()
            mock_model.invoke.return_value = SimpleNamespace(
                content="ok",
                model_copy=lambda update=None: SimpleNamespace(
                    content="ok", id="m", type="ai"
                ),
            )
            mock_provision.return_value = mock_model

            state = {"messages": [HumanMessage(content="hi")], "source_id": "source:xyz"}
            config = {
                "configurable": {
                    "thread_id": SRC_SESSION,
                    "opencode_session_id": SRC_VALUE,
                }
            }
            _call_model_with_source_context_inner(state, config)

        assert mock_provision.await_count == 1
        assert (
            mock_provision.await_args.kwargs["opencode_session_id"] == SRC_VALUE
        )
