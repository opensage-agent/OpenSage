"""Tests for runtime MCP server management.

Covers _bind_ports_to_ip, _allocate_dynamic_port, launch_sandbox,
add_sse_mcp_server, add_stdio_mcp_server, and remove_mcp_server on
OpenSageSandboxManager.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from opensage.config.config_dataclass import (
    ContainerConfig,
    MCPConfig,
    MCPServiceConfig,
    OpenSageConfig,
    SandboxConfig,
)
from opensage.session.opensage_sandbox_manager import OpenSageSandboxManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manager(
    session_id: str = "test-session",
    mcp: Optional[MCPConfig] = None,
    default_host: str = "127.0.0.2",
) -> OpenSageSandboxManager:
    config = OpenSageConfig()
    config.default_host = default_host
    config.sandbox = SandboxConfig()
    if mcp is not None:
        config.mcp = mcp
    session = MagicMock()
    session.opensage_session_id = session_id
    session.config = config
    mgr = OpenSageSandboxManager(session)
    return mgr


# ---------------------------------------------------------------------------
# _bind_ports_to_ip
# ---------------------------------------------------------------------------

class TestBindPortsToIp:
    IP = "127.0.0.5"

    def test_int_binding(self):
        cfg = ContainerConfig(ports={"8080/tcp": 9000})
        OpenSageSandboxManager._bind_ports_to_ip(cfg, self.IP)
        assert cfg.ports["8080/tcp"] == {"host": self.IP, "port": 9000}

    def test_dict_binding(self):
        cfg = ContainerConfig(ports={"8080/tcp": {"port": 9000}})
        OpenSageSandboxManager._bind_ports_to_ip(cfg, self.IP)
        assert cfg.ports["8080/tcp"] == {"host": self.IP, "port": 9000}

    def test_none_binding_unchanged(self):
        cfg = ContainerConfig(ports={"8080/tcp": None})
        OpenSageSandboxManager._bind_ports_to_ip(cfg, self.IP)
        assert cfg.ports["8080/tcp"] is None

    def test_dict_missing_port_raises(self):
        cfg = ContainerConfig(ports={"8080/tcp": {"host": "0.0.0.0"}})
        with pytest.raises(ValueError, match="must contain 'port'"):
            OpenSageSandboxManager._bind_ports_to_ip(cfg, self.IP)

    def test_invalid_type_raises(self):
        cfg = ContainerConfig(ports={"8080/tcp": "bad"})
        with pytest.raises(ValueError, match="str"):
            OpenSageSandboxManager._bind_ports_to_ip(cfg, self.IP)

    def test_empty_ports_noop(self):
        cfg = ContainerConfig(ports={})
        OpenSageSandboxManager._bind_ports_to_ip(cfg, self.IP)
        assert cfg.ports == {}

    def test_no_ports_noop(self):
        cfg = ContainerConfig()
        OpenSageSandboxManager._bind_ports_to_ip(cfg, self.IP)
        assert cfg.ports == {}


# ---------------------------------------------------------------------------
# _allocate_dynamic_port
# ---------------------------------------------------------------------------

class TestAllocateDynamicPort:
    def test_returns_first_available(self):
        from opensage.sandbox.native_docker_sandbox import NativeDockerSandbox

        with patch.object(
            NativeDockerSandbox,
            "_check_port_available",
            side_effect=[False, False, True],
        ):
            port = NativeDockerSandbox._allocate_dynamic_port("127.0.0.1", 9000, 9002)
            assert port == 9002

    def test_exhausted_raises(self):
        from opensage.sandbox.native_docker_sandbox import NativeDockerSandbox

        with patch.object(
            NativeDockerSandbox,
            "_check_port_available",
            return_value=False,
        ):
            with pytest.raises(RuntimeError, match="No available port"):
                NativeDockerSandbox._allocate_dynamic_port("127.0.0.1", 9000, 9001)


# ---------------------------------------------------------------------------
# launch_sandbox
# ---------------------------------------------------------------------------

class TestLaunchSandbox:
    @pytest.mark.asyncio
    async def test_success(self):
        mgr = _make_manager()
        cfg = ContainerConfig(image="test:latest", ports={"80/tcp": 8080})

        mock_sandbox = MagicMock()
        mock_backend = MagicMock()
        mock_backend.create_single_sandbox = AsyncMock(return_value=("test_type", mock_sandbox))
        mock_backend.initialize_all_sandboxes = AsyncMock(return_value={"test_type": None})

        with patch(
            "opensage.session.opensage_sandbox_manager.get_backend_class",
            return_value=mock_backend,
        ):
            result = await mgr.launch_sandbox("test_type", cfg)

        assert result is mock_sandbox
        assert "test_type" in mgr._sandboxes

    @pytest.mark.asyncio
    async def test_duplicate_raises(self):
        mgr = _make_manager()
        mgr._sandboxes["existing"] = MagicMock()

        with pytest.raises(ValueError, match="already exists"):
            await mgr.launch_sandbox("existing", ContainerConfig())

    @pytest.mark.asyncio
    async def test_init_failure_rolls_back(self):
        mgr = _make_manager()
        cfg = ContainerConfig(image="test:latest")

        mock_sandbox = MagicMock()
        mock_backend = MagicMock()
        mock_backend.create_single_sandbox = AsyncMock(return_value=("fail_type", mock_sandbox))
        mock_backend.initialize_all_sandboxes = AsyncMock(
            return_value={"fail_type": RuntimeError("init failed")}
        )

        with patch(
            "opensage.session.opensage_sandbox_manager.get_backend_class",
            return_value=mock_backend,
        ):
            with pytest.raises(RuntimeError, match="init failed"):
                await mgr.launch_sandbox("fail_type", cfg)

        assert "fail_type" not in mgr._sandboxes


# ---------------------------------------------------------------------------
# add_sse_mcp_server
# ---------------------------------------------------------------------------

class TestAddSseMcpServer:
    def test_registers_config(self):
        mgr = _make_manager()
        result = mgr.add_sse_mcp_server(name="my_sse", sse_port=9090, sse_host="127.0.0.2")

        assert result["status"] == "registered"
        assert result["sse_port"] == 9090
        assert "my_sse" in mgr.config.mcp.services

    def test_creates_mcp_config_if_missing(self):
        mgr = _make_manager()
        assert mgr.config.mcp is None

        mgr.add_sse_mcp_server(name="test", sse_port=9090, sse_host="127.0.0.2")

        assert mgr.config.mcp is not None
        assert mgr.config.mcp._parent_config is mgr.config


# ---------------------------------------------------------------------------
# add_stdio_mcp_server
# ---------------------------------------------------------------------------

class TestAddStdioMcpServer:
    @pytest.mark.asyncio
    async def test_success(self):
        mgr = _make_manager()

        with patch(
            "opensage.sandbox.native_docker_sandbox.NativeDockerSandbox._allocate_dynamic_port",
            return_value=9050,
        ), patch.object(mgr, "launch_sandbox", new_callable=AsyncMock) as mock_launch:
            result = await mgr.add_stdio_mcp_server(
                name="my_tool", command="/usr/bin/my-tool", args=["--flag"]
            )

        assert result["status"] == "running"
        assert result["sse_port"] == 9050
        assert result["sandbox_type"] == "mcp_my_tool"

        # Verify ContainerConfig passed to launch_sandbox
        call_args = mock_launch.call_args
        container_cfg = call_args[0][1]
        assert container_cfg.image == "opensage-mcp-proxy:latest"
        assert container_cfg.project_relative_dockerfile_path == "src/opensage/templates/dockerfiles/mcp_stdio_proxy/Dockerfile"
        assert container_cfg.mcp_services == ["my_tool"]
        assert "/usr/bin/my-tool" in container_cfg.command
        assert "--flag" in container_cfg.command

    @pytest.mark.asyncio
    async def test_rollback_on_failure(self):
        mgr = _make_manager()

        with patch(
            "opensage.sandbox.native_docker_sandbox.NativeDockerSandbox._allocate_dynamic_port",
            return_value=9050,
        ), patch.object(
            mgr, "launch_sandbox", new_callable=AsyncMock, side_effect=RuntimeError("boom")
        ):
            with pytest.raises(RuntimeError, match="boom"):
                await mgr.add_stdio_mcp_server(name="failing", command="/bin/fail")

        # MCP config should be cleaned up
        if mgr.config.mcp:
            assert "failing" not in mgr.config.mcp.services

    @pytest.mark.asyncio
    async def test_creates_mcp_config_with_parent(self):
        mgr = _make_manager()
        assert mgr.config.mcp is None

        with patch(
            "opensage.sandbox.native_docker_sandbox.NativeDockerSandbox._allocate_dynamic_port",
            return_value=9050,
        ), patch.object(mgr, "launch_sandbox", new_callable=AsyncMock):
            await mgr.add_stdio_mcp_server(name="tool", command="/bin/tool")

        assert mgr.config.mcp._parent_config is mgr.config


# ---------------------------------------------------------------------------
# remove_mcp_server
# ---------------------------------------------------------------------------

class TestRemoveMcpServer:
    def test_remove_stdio_server(self):
        mgr = _make_manager(
            mcp=MCPConfig(services={"my_tool": MCPServiceConfig(sse_port=9050)}),
        )
        # Simulate an associated sandbox
        mgr._sandboxes["mcp_my_tool"] = MagicMock()

        with patch.object(mgr, "_cleanup_sandbox"):
            result = mgr.remove_mcp_server("my_tool")

        assert result["status"] == "removed"
        assert result["sandbox_removed"] is True
        assert "my_tool" not in mgr.config.mcp.services

    def test_remove_sse_only_server(self):
        mgr = _make_manager(
            mcp=MCPConfig(services={"ext_sse": MCPServiceConfig(sse_port=9090, sse_host="10.0.0.1")}),
        )

        result = mgr.remove_mcp_server("ext_sse")

        assert result["status"] == "removed"
        assert result["sandbox_removed"] is False
        assert "ext_sse" not in mgr.config.mcp.services

    def test_not_found_raises(self):
        mgr = _make_manager()

        with pytest.raises(KeyError, match="not found"):
            mgr.remove_mcp_server("nonexistent")
