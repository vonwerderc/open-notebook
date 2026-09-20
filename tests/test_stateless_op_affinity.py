"""OpenCode session scoping for stateless operations (Task 7).

One value per Ask / transformation / note-title operation, generated at the
outermost boundary and reused by every subcall inside the operation.
"""

import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

SES_RE = re.compile(r"^ses_[0-9a-f]{32}$")


@pytest.fixture
def client():
    from api.main import app

    return TestClient(app)


def _captured_config(mock_graph, call_index=0):
    call = mock_graph.mock_calls[
        [c[0] for c in mock_graph.mock_calls].index("astream") + call_index
    ]
    return call.kwargs["config"]["configurable"]


class TestAskOperations:
    @pytest.mark.asyncio
    @patch("api.routers.search.Model.get", new_callable=AsyncMock)
    @patch("api.routers.search.model_manager.get_embedding_model", new_callable=AsyncMock)
    @patch("api.routers.search.ask_graph")
    async def test_non_streaming_ask_configures_one_operation_value(
        self, mock_graph, mock_embed, mock_model_get, client
    ):
        mock_embed.return_value = SimpleNamespace(id="embedding:1")
        mock_model_get.return_value = SimpleNamespace(id="model:s")

        captured = {}

        async def fake_stream(*args, **kwargs):
            captured["configurable"] = kwargs["config"]["configurable"]
            yield {"write_final_answer": {"final_answer": "42"}}

        mock_graph.astream = MagicMock(side_effect=fake_stream)

        resp = client.post(
            "/api/search/ask/simple",
            json={
                "question": "meaning?",
                "strategy_model": "model:s",
                "answer_model": "model:a",
                "final_answer_model": "model:f",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["answer"] == "42"

        cfg = captured["configurable"]
        value = cfg["opencode_session_id"]
        assert SES_RE.match(value)
        # All three model slots stay distinct from the operation value.
        assert value not in (cfg["strategy_model"], cfg["answer_model"], cfg["final_answer_model"])

    @pytest.mark.asyncio
    @patch("api.routers.search.Model.get", new_callable=AsyncMock)
    @patch("api.routers.search.model_manager.get_embedding_model", new_callable=AsyncMock)
    @patch("api.routers.search.ask_graph")
    async def test_two_ask_operations_differ(
        self, mock_graph, mock_embed, mock_model_get, client
    ):
        mock_embed.return_value = SimpleNamespace(id="embedding:1")
        mock_model_get.return_value = SimpleNamespace(id="model:s")

        captured = []

        async def fake_stream(*args, **kwargs):
            captured.append(kwargs["config"]["configurable"]["opencode_session_id"])
            yield {"write_final_answer": {"final_answer": "42"}}

        mock_graph.astream = MagicMock(side_effect=fake_stream)

        for q in ("a", "b"):
            resp = client.post(
                "/api/search/ask/simple",
                json={
                    "question": q,
                    "strategy_model": "model:s",
                    "answer_model": "model:a",
                    "final_answer_model": "model:f",
                },
            )
            assert resp.status_code == 200

        assert len(captured) == 2
        assert captured[0] != captured[1]


class TestTransformationOperation:
    @pytest.mark.asyncio
    @patch("api.routers.transformations.Model.get", new_callable=AsyncMock)
    @patch("api.routers.transformations.Transformation.get", new_callable=AsyncMock)
    @patch("api.routers.transformations.transformation_graph")
    async def test_execute_passes_one_operation_value(
        self, mock_graph, mock_transformation, mock_model, client
    ):
        mock_transformation.return_value = SimpleNamespace(
            id="transformation:1",
            model_id=None,
            model_dump=lambda: {"id": "transformation:1"},
        )
        mock_model.return_value = SimpleNamespace(id="model:1")
        mock_graph.ainvoke = AsyncMock(return_value={"output": "ok"})

        resp = client.post(
            "/api/transformations/execute",
            json={"transformation_id": "transformation:1", "input_text": "text"},
        )
        assert resp.status_code == 200
        _, kwargs = mock_graph.ainvoke.call_args
        assert SES_RE.match(kwargs["config"]["configurable"]["opencode_session_id"])


class TestNoteTitleOperation:
    @pytest.mark.asyncio
    @patch("open_notebook.domain.notebook.Note.save", new_callable=AsyncMock)
    @patch("open_notebook.graphs.prompt.graph")
    async def test_ai_note_title_gets_operation_value(self, mock_graph, mock_save, client):
        mock_save.return_value = "note:1"
        mock_graph.ainvoke = AsyncMock(return_value={"output": "A Title"})

        resp = client.post(
            "/api/notes",
            json={"content": "some content", "note_type": "ai"},
        )
        assert resp.status_code == 200
        _, kwargs = mock_graph.ainvoke.call_args
        assert SES_RE.match(
            kwargs["config"]["configurable"]["opencode_session_id"]
        )


class TestGraphNodesUseConfiguredValue:
    @pytest.mark.asyncio
    async def test_ask_nodes_forward_configured_value(self):
        from open_notebook.graphs import ask as ask_graph_module

        captured = []

        class FakeModel:
            async def ainvoke(self, payload):
                from types import SimpleNamespace as NS

                return NS(content='{"reasoning": "r", "searches": []}')

        async def fake_provision(content, model_id, default_type, **kwargs):
            captured.append(kwargs.get("opencode_session_id"))
            return FakeModel()

        config = {
            "configurable": {
                "strategy_model": "model:s",
                "answer_model": "model:a",
                "final_answer_model": "model:f",
                "opencode_session_id": "ses_fixed",
            }
        }
        with patch.object(ask_graph_module, "provision_langchain_model", fake_provision), patch.object(
            ask_graph_module, "vector_search", new_callable=AsyncMock, return_value=[{"id": "1", "content": "c"}]
        ):
            await ask_graph_module.call_model_with_messages(
                {"question": "q"}, config  # type: ignore[typeddict-item,arg-type]
            )
            await ask_graph_module.provide_answer(
                {"question": "q", "term": "t", "instructions": "i"},  # type: ignore[typeddict-item]
                config,  # type: ignore[arg-type]
            )
            await ask_graph_module.write_final_answer(
                {"question": "q"}, config  # type: ignore[typeddict-item,arg-type]
            )

        assert captured == ["ses_fixed", "ses_fixed", "ses_fixed"]

    @pytest.mark.asyncio
    async def test_transformation_node_forwards_configured_value(self):
        from open_notebook.graphs import transformation as tx_module

        captured = []

        class FakeModel:
            async def ainvoke(self, payload):
                from types import SimpleNamespace as NS

                return NS(content="out")

        async def fake_provision(content, model_id, default_type, **kwargs):
            captured.append(kwargs.get("opencode_session_id"))
            return FakeModel()

        state = {
            "input_text": "hello",
            "transformation": SimpleNamespace(
                prompt="p", title="t", name="n", model_id="model:1"
            ),
        }
        config = {
            "configurable": {"model_id": "model:1", "opencode_session_id": "ses_op"}
        }
        with patch.object(tx_module, "provision_langchain_model", fake_provision):
            await tx_module.run_transformation(
                state, config  # type: ignore[arg-type]
            )

        assert captured == ["ses_op"]

    @pytest.mark.asyncio
    async def test_prompt_node_forwards_configured_value(self):
        from open_notebook.graphs import prompt as prompt_module

        captured = []

        class FakeModel:
            async def ainvoke(self, payload):
                from types import SimpleNamespace as NS

                return NS(content="title")

        async def fake_provision(content, model_id, default_type, **kwargs):
            captured.append(kwargs.get("opencode_session_id"))
            return FakeModel()

        config = {"configurable": {"opencode_session_id": "ses_title"}}
        with patch.object(prompt_module, "provision_langchain_model", fake_provision):
            await prompt_module.call_model(
                {"input_text": "text", "prompt": "make a title"},
                config,  # type: ignore[arg-type]
            )

        assert captured == ["ses_title"]
