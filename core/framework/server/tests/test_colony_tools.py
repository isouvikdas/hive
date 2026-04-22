"""Tests for the per-colony MCP tool allowlist filter + routes.

Covers:
1. ``ColonyRuntime`` filter semantics (default-allow, allowlist, empty,
   lifecycle passes through).
2. routes_colony_tools round trip (GET/PATCH, validation, 404).
3. Colony index route for the Tool Library picker.

Routes never touch the real ``~/.hive/colonies`` tree — we redirect
``COLONIES_DIR`` into ``tmp_path`` via monkeypatch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from framework.host.colony_runtime import ColonyRuntime
from framework.llm.provider import Tool
from framework.server import routes_colony_tools


def _tool(name: str) -> Tool:
    return Tool(name=name, description=f"desc of {name}", parameters={"type": "object"})


# ---------------------------------------------------------------------------
# ColonyRuntime filter unit tests
# ---------------------------------------------------------------------------


def _bare_runtime() -> ColonyRuntime:
    rt = ColonyRuntime.__new__(ColonyRuntime)
    rt._enabled_mcp_tools = None
    rt._mcp_tool_names_all = set()
    return rt


class TestColonyFilter:
    def test_default_is_noop(self):
        rt = _bare_runtime()
        tools = [_tool("mcp_a"), _tool("lc_b")]
        assert rt._apply_tool_allowlist(tools) == tools

    def test_allowlist_gates_mcp_only(self):
        rt = _bare_runtime()
        rt._mcp_tool_names_all = {"mcp_a", "mcp_b"}
        rt._enabled_mcp_tools = ["mcp_a"]
        tools = [_tool("mcp_a"), _tool("mcp_b"), _tool("lc_c")]
        names = [t.name for t in rt._apply_tool_allowlist(tools)]
        assert names == ["mcp_a", "lc_c"]

    def test_empty_allowlist_keeps_lifecycle(self):
        rt = _bare_runtime()
        rt._mcp_tool_names_all = {"mcp_a", "mcp_b"}
        rt._enabled_mcp_tools = []
        tools = [_tool("mcp_a"), _tool("mcp_b"), _tool("lc_c")]
        names = [t.name for t in rt._apply_tool_allowlist(tools)]
        assert names == ["lc_c"]

    def test_setter_mutates_live_state(self):
        rt = _bare_runtime()
        rt.set_tool_allowlist(["x"], {"x", "y"})
        assert rt._enabled_mcp_tools == ["x"]
        assert rt._mcp_tool_names_all == {"x", "y"}

        # Passing None on allowlist clears gating; mcp_tool_names_all
        # defaults to "keep current" so a subsequent caller doesn't need
        # to repeat the set.
        rt.set_tool_allowlist(None)
        assert rt._enabled_mcp_tools is None
        assert rt._mcp_tool_names_all == {"x", "y"}


# ---------------------------------------------------------------------------
# Route round-trip tests
# ---------------------------------------------------------------------------


@dataclass
class _FakeSession:
    colony_name: str
    colony: Any = None
    colony_runtime: Any = None
    id: str = "sess-1"


@dataclass
class _FakeManager:
    _sessions: dict = field(default_factory=dict)
    _mcp_tool_catalog: dict = field(default_factory=dict)


@pytest.fixture
def colony_dir(tmp_path, monkeypatch):
    """Point COLONIES_DIR into a tmp tree and seed a colony."""
    colonies = tmp_path / "colonies"
    colonies.mkdir()
    monkeypatch.setattr("framework.host.colony_metadata.COLONIES_DIR", colonies)

    name = "my_colony"
    cdir = colonies / name
    cdir.mkdir()
    (cdir / "metadata.json").write_text(
        json.dumps(
            {
                "colony_name": name,
                "queen_name": "queen_technology",
                "created_at": "2026-04-20T00:00:00+00:00",
            }
        )
    )
    return colonies, name


async def _app(manager: _FakeManager) -> web.Application:
    app = web.Application()
    app["manager"] = manager
    routes_colony_tools.register_routes(app)
    return app


@pytest.mark.asyncio
async def test_get_tools_default_allow(colony_dir):
    _, name = colony_dir
    manager = _FakeManager(
        _mcp_tool_catalog={
            "coder-tools": [
                {"name": "read_file", "description": "read", "input_schema": {}},
                {"name": "write_file", "description": "write", "input_schema": {}},
            ],
        }
    )
    app = await _app(manager)
    async with TestClient(TestServer(app)) as client:
        resp = await client.get(f"/api/colony/{name}/tools")
        assert resp.status == 200
        body = await resp.json()
    assert body["enabled_mcp_tools"] is None
    assert body["stale"] is False
    tools = {t["name"]: t for t in body["mcp_servers"][0]["tools"]}
    assert all(t["enabled"] for t in tools.values())


@pytest.mark.asyncio
async def test_patch_persists_and_validates(colony_dir):
    colonies_dir, name = colony_dir
    manager = _FakeManager(
        _mcp_tool_catalog={
            "coder-tools": [
                {"name": "read_file", "description": "", "input_schema": {}},
                {"name": "write_file", "description": "", "input_schema": {}},
            ]
        }
    )
    app = await _app(manager)
    async with TestClient(TestServer(app)) as client:
        resp = await client.patch(
            f"/api/colony/{name}/tools", json={"enabled_mcp_tools": ["read_file"]}
        )
        assert resp.status == 200
        body = await resp.json()
        assert body["enabled_mcp_tools"] == ["read_file"]

        # Persisted to metadata.json
        raw = json.loads((colonies_dir / name / "metadata.json").read_text())
        assert raw["enabled_mcp_tools"] == ["read_file"]

        # GET reflects the allowlist
        resp = await client.get(f"/api/colony/{name}/tools")
        body = await resp.json()
        tools = {t["name"]: t for t in body["mcp_servers"][0]["tools"]}
        assert tools["read_file"]["enabled"] is True
        assert tools["write_file"]["enabled"] is False

        # Unknown → 400
        resp = await client.patch(
            f"/api/colony/{name}/tools", json={"enabled_mcp_tools": ["ghost"]}
        )
        assert resp.status == 400
        assert "ghost" in (await resp.json()).get("unknown", [])


@pytest.mark.asyncio
async def test_patch_refreshes_live_runtime(colony_dir):
    _, name = colony_dir

    rt = _bare_runtime()
    rt._mcp_tool_names_all = {"read_file", "write_file"}
    rt.set_tool_allowlist(None)

    session = _FakeSession(colony_name=name, colony=rt)
    manager = _FakeManager(
        _sessions={session.id: session},
        _mcp_tool_catalog={
            "coder-tools": [
                {"name": "read_file", "description": "", "input_schema": {}},
                {"name": "write_file", "description": "", "input_schema": {}},
            ]
        },
    )

    app = await _app(manager)
    async with TestClient(TestServer(app)) as client:
        resp = await client.patch(
            f"/api/colony/{name}/tools", json={"enabled_mcp_tools": ["read_file"]}
        )
        assert resp.status == 200
        body = await resp.json()
        assert body["refreshed_runtimes"] == 1
    assert rt._enabled_mcp_tools == ["read_file"]


@pytest.mark.asyncio
async def test_404_for_unknown_colony(colony_dir):
    manager = _FakeManager()
    app = await _app(manager)
    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/api/colony/unknown/tools")
        assert resp.status == 404
        resp = await client.patch(
            "/api/colony/unknown/tools", json={"enabled_mcp_tools": None}
        )
        assert resp.status == 404


@pytest.mark.asyncio
async def test_tools_index_lists_colonies(colony_dir):
    _, name = colony_dir
    manager = _FakeManager()
    app = await _app(manager)
    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/api/colonies/tools-index")
        assert resp.status == 200
        body = await resp.json()
    entries = {c["name"]: c for c in body["colonies"]}
    assert name in entries
    assert entries[name]["queen_name"] == "queen_technology"
    assert entries[name]["has_allowlist"] is False
