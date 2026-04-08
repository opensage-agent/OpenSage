"""Unit tests for create_subagent dynamic tool behavior."""

from __future__ import annotations

import types as _types

import pytest

from opensage.toolbox.general import dynamic_subagent as dyn


class _DummyAgent:
    def __init__(self, name: str):
        self.name = name


class _DummyInvocationContext:
    def __init__(self, agent_name: str):
        self.agent = _DummyAgent(agent_name)


class _DummyToolContext:
    def __init__(self, agent_name: str = "caller"):
        self._invocation_context = _DummyInvocationContext(agent_name)


class _DummyEnsemble:
    def __init__(self, models: list[str]):
        self._models = models

    def get_available_models(self):
        return self._models


class _CapturingAgentManager:
    def __init__(self):
        self.last_config = None

    async def create_agent(self, config, creator=None, persist=True):
        self.last_config = config
        # Return a dummy agent instance; caller only needs agent_id + instance.
        return "agent-id", _types.SimpleNamespace(
            instruction=config.get("instruction", "")
        )

    async def update_agent_status(self, agent_id, status):
        return True


class _DummySandboxManager:
    """Minimal sandbox manager stub for runtime MCP toolset tests."""

    def __init__(self, mcp_toolsets: dict | None = None):
        self._mcp_toolsets = mcp_toolsets or {}

    def get_runtime_mcp_toolset(self, name: str):
        return self._mcp_toolsets.get(name)


class _DummySession:
    def __init__(self, models: list[str], mcp_toolsets: dict | None = None):
        self.agents = _CapturingAgentManager()
        self.ensemble = _DummyEnsemble(models=models)
        self.sandboxes = _DummySandboxManager(mcp_toolsets=mcp_toolsets)


@pytest.mark.asyncio
async def test_create_subagent_rejects_empty_tools_list(monkeypatch):
    tool_context = _DummyToolContext()
    session = _DummySession(models=["openai/gpt-5"])

    monkeypatch.setattr(dyn, "get_opensage_session_id_from_context", lambda tc: "sid")
    monkeypatch.setattr(dyn, "get_opensage_session", lambda sid: session)
    monkeypatch.setattr(dyn, "extract_tools_from_agent", lambda agent: {"x": object()})

    res = await dyn.create_subagent(
        agent_name="a",
        instruction="do stuff",
        model_name="openai/gpt-5",
        tools_list=[],
        enabled_skills=[],
        tool_context=tool_context,
    )

    assert res["success"] is False
    assert "tools_list must not be empty" in res["error"]


@pytest.mark.asyncio
async def test_create_subagent_injects_default_tools_and_adds_skills_guardrail(
    monkeypatch,
):
    tool_context = _DummyToolContext()
    session = _DummySession(models=["openai/gpt-5"])

    monkeypatch.setattr(dyn, "get_opensage_session_id_from_context", lambda tc: "sid")
    monkeypatch.setattr(dyn, "get_opensage_session", lambda sid: session)

    extra_tool_obj = object()
    monkeypatch.setattr(
        dyn, "extract_tools_from_agent", lambda agent: {"extra_tool": extra_tool_obj}
    )

    res = await dyn.create_subagent(
        agent_name="a",
        instruction="do stuff",
        model_name="openai/gpt-5",
        tools_list=["extra_tool"],
        enabled_skills=["coverage"],
        tool_context=tool_context,
    )
    assert res["success"] is True

    cfg = session.agents.last_config
    assert isinstance(cfg, dict)

    # Default baseline tool names must always be included.
    assert "run_terminal_command" in cfg["tool_names"]
    assert "list_background_tasks" in cfg["tool_names"]
    assert "get_background_task_output" in cfg["tool_names"]
    assert "complain" in cfg["tool_names"]
    assert "extra_tool" in cfg["tool_names"]

    # Default baseline tool objects must always be included.
    assert dyn.run_terminal_command in cfg["tools"]
    assert dyn.list_background_tasks in cfg["tools"]
    assert dyn.get_background_task_output in cfg["tools"]
    assert dyn.complain in cfg["tools"]
    assert extra_tool_obj in cfg["tools"]

    # Instruction should only restrict bash tools by enabled_skills (no "only these tools").
    instr = cfg["instruction"]
    assert "enabled_skills" in instr
    assert "must only use bash tools" in instr
    assert "only use these tools" not in instr.lower()


@pytest.mark.asyncio
async def test_create_subagent_inherit_model_passes_resolved_model(monkeypatch):
    tool_context = _DummyToolContext()
    # The available models list does not necessarily include the special
    # sentinel model name "inherit". create_subagent should still allow it.
    session = _DummySession(models=["openai/gpt-5"])

    monkeypatch.setattr(dyn, "get_opensage_session_id_from_context", lambda tc: "sid")
    monkeypatch.setattr(dyn, "get_opensage_session", lambda sid: session)
    monkeypatch.setattr(dyn, "extract_tools_from_agent", lambda agent: {"t": object()})
    monkeypatch.setattr(dyn, "get_model_from_agent", lambda agent: "MODEL_OBJ")

    res = await dyn.create_subagent(
        agent_name="a",
        instruction="do stuff",
        model_name="inherit",
        tools_list=["t"],
        enabled_skills=[],
        tool_context=tool_context,
    )
    assert res["success"] is True

    cfg = session.agents.last_config
    assert cfg["model"] == "inherit"
    assert cfg["_resolved_model"] == "MODEL_OBJ"


# ---------------------------------------------------------------------------
# Runtime MCP toolset fallback in create_subagent
# ---------------------------------------------------------------------------


class _FakeMCPToolset:
    """Minimal stand-in for OpenSageMCPToolset."""

    def __init__(self, name: str):
        self.name = name
        self.tool_name_prefix = name


@pytest.mark.asyncio
async def test_create_subagent_resolves_runtime_mcp_tool(monkeypatch):
    """Tool name not in caller's tools but in config.mcp.services resolves."""
    tool_context = _DummyToolContext()
    fake_toolset = _FakeMCPToolset("my_runtime_mcp")
    session = _DummySession(
        models=["openai/gpt-5"],
        mcp_toolsets={"my_runtime_mcp": fake_toolset},
    )

    monkeypatch.setattr(dyn, "get_opensage_session_id_from_context", lambda tc: "sid")
    monkeypatch.setattr(dyn, "get_opensage_session", lambda sid: session)
    # Caller agent has a dummy tool (needed to pass the empty-tools guard)
    # but NOT the runtime MCP tool.
    dummy = lambda: None
    dummy.__name__ = "some_other_tool"
    monkeypatch.setattr(dyn, "extract_tools_from_agent", lambda agent: {"some_other_tool": dummy})

    res = await dyn.create_subagent(
        agent_name="mcp_user",
        instruction="use the tool",
        model_name="openai/gpt-5",
        tools_list=["my_runtime_mcp"],
        enabled_skills=[],
        tool_context=tool_context,
    )

    assert res["success"] is True
    cfg = session.agents.last_config
    assert fake_toolset in cfg["tools"]
    assert "my_runtime_mcp" in cfg["tool_names"]


@pytest.mark.asyncio
async def test_create_subagent_runtime_mcp_still_rejects_unknown(monkeypatch):
    """Tool name not in caller's tools AND not in MCP services is still invalid."""
    tool_context = _DummyToolContext()
    session = _DummySession(models=["openai/gpt-5"])  # no mcp_toolsets

    monkeypatch.setattr(dyn, "get_opensage_session_id_from_context", lambda tc: "sid")
    monkeypatch.setattr(dyn, "get_opensage_session", lambda sid: session)
    dummy = lambda: None
    dummy.__name__ = "some_tool"
    monkeypatch.setattr(dyn, "extract_tools_from_agent", lambda agent: {"some_tool": dummy})

    res = await dyn.create_subagent(
        agent_name="a",
        instruction="do stuff",
        model_name="openai/gpt-5",
        tools_list=["nonexistent_tool"],
        enabled_skills=[],
        tool_context=tool_context,
    )

    assert res["success"] is False
    assert "nonexistent_tool" in res["error"]
