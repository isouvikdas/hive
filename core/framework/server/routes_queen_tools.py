"""Per-queen MCP tool allowlist routes.

- GET   /api/queen/{queen_id}/tools  -- enumerate the queen's tool surface
- PATCH /api/queen/{queen_id}/tools  -- set or clear the MCP tool allowlist

Lifecycle and synthetic tools (``ask_user``) are always part of the queen's
surface in INDEPENDENT mode and are returned with ``editable: false``. MCP
tools are grouped by origin server and carry per-tool ``enabled`` flags.

The allowlist is a persisted queen-profile field, ``enabled_mcp_tools``:

- ``null`` / missing  -> "allow every MCP tool" (default, backward-compat)
- ``[]``              -> explicitly disable every MCP tool
- ``["foo", "bar"]``  -> only these MCP tools pass through to the LLM

Filtering happens in ``QueenPhaseState.rebuild_independent_filter`` so the
LLM prompt cache stays warm between saves.
"""

from __future__ import annotations

import logging
from typing import Any

from aiohttp import web

from framework.agents.queen.queen_profiles import (
    ensure_default_queens,
    load_queen_profile,
    update_queen_profile,
)

logger = logging.getLogger(__name__)


_SYNTHETIC_NAMES = {"ask_user"}


async def _ensure_manager_catalog(manager: Any) -> dict[str, list[dict[str, Any]]]:
    """Return the cached MCP tool catalog, building it on first call.

    ``queen_orchestrator.create_queen`` populates ``_mcp_tool_catalog`` on
    every queen boot. On a fresh backend process the user may open the
    Tool Library before any queen session has started, so the catalog is
    empty. In that case we build one from the shared MCP config; the
    first call pays an MCP-subprocess-spawn cost, subsequent calls are
    cache hits. The build runs off the event loop via asyncio.to_thread
    so the HTTP worker stays responsive while MCP servers initialize.
    """
    if manager is None:
        return {}
    catalog = getattr(manager, "_mcp_tool_catalog", None)
    if isinstance(catalog, dict) and catalog:
        return catalog
    try:
        import asyncio

        from framework.server.queen_orchestrator import build_queen_tool_registry_bare

        registry, built = await asyncio.to_thread(build_queen_tool_registry_bare)
        manager._mcp_tool_catalog = built  # type: ignore[attr-defined]
        manager._bootstrap_tool_registry = registry  # type: ignore[attr-defined]
        return built
    except Exception:
        logger.warning("Tool catalog bootstrap failed", exc_info=True)
        return {}


def _lifecycle_entries_without_session(
    manager: Any,
    mcp_names: set[str],
) -> list[dict[str, Any]]:
    """Derive lifecycle tool names from the registry even without a session.

    We register queen lifecycle tools against a temporary registry using a
    minimal stub, then subtract the MCP-origin set and the synthetic set.
    The result matches what the queen sees at runtime (minus context-
    specific variants).
    """
    registry = getattr(manager, "_bootstrap_tool_registry", None)
    # If the bootstrap registry exists but doesn't carry lifecycle tools
    # yet, register them now.
    if registry is not None and not getattr(registry, "_lifecycle_bootstrap_done", False):
        try:
            from types import SimpleNamespace

            from framework.tools.queen_lifecycle_tools import register_queen_lifecycle_tools

            stub_session = SimpleNamespace(
                id="tool-library-bootstrap",
                colony_runtime=None,
                event_bus=None,
                worker_path=None,
                phase_state=None,
                llm=None,
            )
            register_queen_lifecycle_tools(
                registry,
                session=stub_session,
                session_id=stub_session.id,
                session_manager=None,
                manager_session_id=stub_session.id,
                phase_state=None,
            )
            registry._lifecycle_bootstrap_done = True  # type: ignore[attr-defined]
        except Exception:
            logger.debug("lifecycle bootstrap failed", exc_info=True)

    if registry is None:
        return []

    out: list[dict[str, Any]] = []
    for name, tool in sorted(registry.get_tools().items()):
        if name in mcp_names or name in _SYNTHETIC_NAMES:
            continue
        out.append(
            {
                "name": tool.name,
                "description": tool.description,
                "editable": False,
            }
        )
    return out


def _synthetic_entries() -> list[dict[str, Any]]:
    """Return display metadata for synthetic tools injected by the agent loop.

    Kept behind a lazy import so test harnesses that don't wire the agent
    loop can still hit this route without blowing up.
    """
    try:
        from framework.agent_loop.internals.synthetic_tools import build_ask_user_tool

        tool = build_ask_user_tool()
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "editable": False,
            }
        ]
    except Exception:
        return [
            {
                "name": "ask_user",
                "description": "Pause and ask the user a structured question.",
                "editable": False,
            }
        ]


def _live_queen_session(manager: Any, queen_id: str) -> Any:
    """Return any live DM session owned by this queen, or ``None``."""
    sessions = getattr(manager, "_sessions", None) or {}
    for session in sessions.values():
        if getattr(session, "queen_name", None) != queen_id:
            continue
        # Prefer DM (non-colony) sessions
        if getattr(session, "colony_runtime", None) is None:
            return session
    return None


def _render_mcp_servers(
    *,
    mcp_tool_names_by_server: dict[str, list[dict[str, Any]]],
    enabled_mcp_tools: list[str] | None,
) -> list[dict[str, Any]]:
    """Shape the mcp_tool_catalog entries for the API response."""
    allowed: set[str] | None = None if enabled_mcp_tools is None else set(enabled_mcp_tools)
    servers: list[dict[str, Any]] = []
    for server_name in sorted(mcp_tool_names_by_server):
        entries = mcp_tool_names_by_server[server_name]
        tools = []
        for entry in entries:
            name = entry.get("name")
            enabled = True if allowed is None else name in allowed
            tools.append(
                {
                    "name": name,
                    "description": entry.get("description", ""),
                    "input_schema": entry.get("input_schema", {}),
                    "enabled": enabled,
                }
            )
        servers.append({"name": server_name, "tools": tools})
    return servers


def _catalog_from_live_session(session: Any) -> dict[str, list[dict[str, Any]]]:
    """Rebuild a per-server tool catalog from a live queen session.

    The session's registry is authoritative — this reflects any hot-added
    MCP servers since the manager-level snapshot was cached.
    """
    registry = getattr(session, "_queen_tool_registry", None)
    if registry is None:
        # session._queen_tools_by_name is a stash from create_queen; we
        # only have registry via the tools list, so reconstruct from the
        # phase state instead.
        phase_state = getattr(session, "phase_state", None)
        if phase_state is None:
            return {}
        mcp_names = getattr(phase_state, "mcp_tool_names_all", set()) or set()
        independent_tools = getattr(phase_state, "independent_tools", []) or []
        result: dict[str, list[dict[str, Any]]] = {"(unknown)": []}
        for tool in independent_tools:
            if tool.name not in mcp_names:
                continue
            result["(unknown)"].append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.parameters,
                }
            )
        return result if result["(unknown)"] else {}

    server_map = getattr(registry, "_mcp_server_tools", {}) or {}
    tools_by_name = {t.name: t for t in registry.get_tools().values()}
    catalog: dict[str, list[dict[str, Any]]] = {}
    for server_name, tool_names in server_map.items():
        entries: list[dict[str, Any]] = []
        for name in sorted(tool_names):
            tool = tools_by_name.get(name)
            if tool is None:
                continue
            entries.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.parameters,
                }
            )
        catalog[server_name] = entries
    return catalog


def _lifecycle_entries(
    *,
    session: Any,
    mcp_tool_names_all: set[str],
) -> list[dict[str, Any]]:
    """Lifecycle tools = independent_tools minus MCP-origin minus synthetic.

    We compute this from a live session when available so the list exactly
    matches what the queen actually sees on her next turn.
    """
    if session is None:
        return []
    phase_state = getattr(session, "phase_state", None)
    if phase_state is None:
        return []
    result: list[dict[str, Any]] = []
    for tool in getattr(phase_state, "independent_tools", []) or []:
        if tool.name in mcp_tool_names_all:
            continue
        if tool.name in _SYNTHETIC_NAMES:
            continue
        result.append(
            {
                "name": tool.name,
                "description": tool.description,
                "editable": False,
            }
        )
    return sorted(result, key=lambda x: x["name"])


async def handle_get_tools(request: web.Request) -> web.Response:
    """GET /api/queen/{queen_id}/tools — enumerate tool surface for the UI."""
    queen_id = request.match_info["queen_id"]
    ensure_default_queens()
    try:
        profile = load_queen_profile(queen_id)
    except FileNotFoundError:
        return web.json_response({"error": f"Queen '{queen_id}' not found"}, status=404)

    manager = request.app.get("manager")
    session = _live_queen_session(manager, queen_id) if manager is not None else None

    # Prefer a live session's registry for freshness. Otherwise use (or
    # build on demand) the manager-level catalog so the Tool Library works
    # even before any queen has been started in this process.
    if session is not None:
        catalog = _catalog_from_live_session(session)
    else:
        catalog = await _ensure_manager_catalog(manager)
    stale = not catalog

    mcp_tool_names_all: set[str] = set()
    for entries in catalog.values():
        for entry in entries:
            if entry.get("name"):
                mcp_tool_names_all.add(entry["name"])

    if session is not None:
        lifecycle = _lifecycle_entries(
            session=session,
            mcp_tool_names_all=mcp_tool_names_all,
        )
    else:
        lifecycle = _lifecycle_entries_without_session(manager, mcp_tool_names_all)

    enabled_mcp_tools = profile.get("enabled_mcp_tools")

    response = {
        "queen_id": queen_id,
        "enabled_mcp_tools": enabled_mcp_tools,
        "stale": stale,
        "lifecycle": lifecycle,
        "synthetic": _synthetic_entries(),
        "mcp_servers": _render_mcp_servers(
            mcp_tool_names_by_server=catalog,
            enabled_mcp_tools=enabled_mcp_tools,
        ),
    }
    return web.json_response(response)


async def handle_patch_tools(request: web.Request) -> web.Response:
    """PATCH /api/queen/{queen_id}/tools — persist the MCP tool allowlist.

    Body: ``{"enabled_mcp_tools": null | string[]}``.

    - ``null`` resets to "allow every MCP tool" (default).
    - A list is validated against the known MCP catalog; unknown names
      are rejected with 400 so the frontend catches typos.
    """
    queen_id = request.match_info["queen_id"]
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)
    if not isinstance(body, dict) or "enabled_mcp_tools" not in body:
        return web.json_response(
            {"error": "Body must be an object with an 'enabled_mcp_tools' field"},
            status=400,
        )

    enabled = body["enabled_mcp_tools"]
    if enabled is not None:
        if not isinstance(enabled, list) or not all(isinstance(x, str) for x in enabled):
            return web.json_response(
                {"error": "'enabled_mcp_tools' must be null or a list of strings"},
                status=400,
            )

    ensure_default_queens()
    try:
        load_queen_profile(queen_id)
    except FileNotFoundError:
        return web.json_response({"error": f"Queen '{queen_id}' not found"}, status=404)

    # Validate names against the known MCP tool catalog. We prefer a live
    # session's registry for the most up-to-date set, then fall back to
    # the manager-level snapshot (building it on demand if absent).
    manager = request.app.get("manager")
    session = _live_queen_session(manager, queen_id) if manager is not None else None
    if session is not None:
        catalog = _catalog_from_live_session(session)
    else:
        catalog = await _ensure_manager_catalog(manager)
    known_names: set[str] = set()
    for entries in catalog.values():
        for entry in entries:
            if entry.get("name"):
                known_names.add(entry["name"])

    if enabled is not None and known_names:
        unknown = sorted(set(enabled) - known_names)
        if unknown:
            return web.json_response(
                {"error": "Unknown MCP tool name(s)", "unknown": unknown},
                status=400,
            )

    # Persist — we pass the raw value (``None`` → stored as YAML null).
    updated = update_queen_profile(queen_id, {"enabled_mcp_tools": enabled})

    # Hot-reload every live DM session for this queen. The filter memo is
    # rebuilt so the very next turn sees the new allowlist without a
    # session restart, and the prompt cache is invalidated exactly once.
    refreshed = 0
    sessions = getattr(manager, "_sessions", None) or {}
    for sess in sessions.values():
        if getattr(sess, "queen_name", None) != queen_id:
            continue
        phase_state = getattr(sess, "phase_state", None)
        if phase_state is None:
            continue
        phase_state.enabled_mcp_tools = enabled
        rebuild = getattr(phase_state, "rebuild_independent_filter", None)
        if callable(rebuild):
            try:
                rebuild()
                refreshed += 1
            except Exception:
                logger.debug(
                    "Queen tools: rebuild_independent_filter failed for session %s",
                    getattr(sess, "id", "?"),
                    exc_info=True,
                )

    logger.info(
        "Queen tools: queen_id=%s allowlist=%s refreshed_sessions=%d",
        queen_id,
        "null" if enabled is None else f"{len(enabled)} tool(s)",
        refreshed,
    )
    return web.json_response(
        {
            "queen_id": queen_id,
            "enabled_mcp_tools": updated.get("enabled_mcp_tools"),
            "refreshed_sessions": refreshed,
        }
    )


def register_routes(app: web.Application) -> None:
    """Register queen-tools routes."""
    app.router.add_get("/api/queen/{queen_id}/tools", handle_get_tools)
    app.router.add_patch("/api/queen/{queen_id}/tools", handle_patch_tools)
