"""Role-based default tool allowlists for queens.

Every queen inherits the same MCP surface (all servers loaded for the
queen agent), but exposing 94+ tools to every persona clutters the LLM
tool catalog and wastes prompt tokens. This module defines a sensible
default allowlist per queen persona so, e.g., Head of Legal doesn't
see port scanners and Head of Finance doesn't see ``apply_patch``.

Defaults apply only when the queen has no ``tools.json`` sidecar — the
moment the user saves an allowlist through the Tool Library, the
sidecar becomes authoritative. A DELETE on the tools endpoint removes
the sidecar and brings the queen back to her role default.

Category entries support a ``@server:NAME`` shorthand that expands to
every tool name registered against that MCP server in the current
catalog. This keeps the category table short and drift-free when new
tools are added (e.g. browser_* auto-joins the ``browser`` category).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Categories — reusable bundles of MCP tool names.
# ---------------------------------------------------------------------------
#
# Each category is a flat list of either concrete tool names or the
# ``@server:NAME`` shorthand. The shorthand expands to every tool the
# given MCP server currently exposes (requires a live catalog; when one
# is not available the shorthand is silently dropped so we fall back to
# the named entries only).

_TOOL_CATEGORIES: dict[str, list[str]] = {
    # Unified file ops — read, write, edit, search across the post-refactor
    # files-tools MCP server (read_file, write_file, edit_file, hashline_edit,
    # apply_patch, search_files). 
    "file_ops": [
        "@server:files-tools",
    ],
    # Terminal + process control — engineering personas only.
    # The terminal-tools MCP server covers foreground exec with auto-promotion,
    # background jobs, persistent PTY sessions, and ripgrep/find search.
    "terminal": [
        "@server:terminal-tools",
    ],
    # Tabular data. CSV/Excel read/write + DuckDB SQL.
    "advanced_spreadsheet": [
        "csv_read",
        "csv_info",
        "csv_write",
        "csv_append",
        "csv_sql",
        "excel_read",
        "excel_info",
        "excel_write",
        "excel_append",
        "excel_search",
        "excel_sheet_list",
        "excel_sql",
    ],
    # Browser lifecycle + read-only inspection (navigation, snapshots, query).
    # Split out from interaction so personas that only need to *observe* pages
    # (e.g. research, status checks) don't pull in click/type/drag/etc.
    "browser_basic": [
        "browser_setup",
        "browser_status",
        "browser_start",
        "browser_stop",
        "browser_tabs",
        "browser_open",
        "browser_close",
        "browser_activate_tab",
        "browser_close_all",
        "browser_close_finished",
        "browser_navigate",
        "browser_go_back",
        "browser_go_forward",
        "browser_reload",
        "browser_screenshot",
        "browser_snapshot",
        "browser_html",
        "browser_console",
        "browser_evaluate",
        "browser_get_text",
        "browser_get_attribute",
        "browser_get_rect",
        "browser_shadow_query",
    ],
    # Browser interaction — anything that mutates page state (clicks, typing,
    # drag, scrolling, dialogs, file uploads). Pair with browser_basic for
    # full automation; omit for read-only personas.
    "browser_interaction": [
        "browser_click",
        "browser_click_coordinate",
        "browser_type",
        "browser_fill",
        "browser_type_focused",
        "browser_press",
        "browser_press_at",
        "browser_hover",
        "browser_hover_coordinate",
        "browser_select",
        "browser_scroll",
        "browser_drag",
        "browser_wait",
        "browser_resize",
        "browser_upload",
        "browser_dialog",
    ],
    # Lightweight context helpers — good default for every queen.
    "time_context": [
        "get_current_time",
        "get_account_info",
    ]
}


# ---------------------------------------------------------------------------
# Per-queen mapping.
# ---------------------------------------------------------------------------
#
# Built from the queen personas in ``queen_profiles.DEFAULT_QUEENS``. The
# goal is "just enough" — a queen should see tools she'd plausibly call
# for her stated role, nothing more. Users curate further via the Tool
# Library if they want.
#
# A queen whose ID is NOT in this map falls through to "allow every MCP
# tool" (the original behavior), which keeps the system compatible with
# user-added custom queen IDs that we don't know about.

QUEEN_DEFAULT_CATEGORIES: dict[str, list[str]] = {
    # Head of Technology — builds and operates systems; full toolkit.
    "queen_technology": [
        "file_ops",
        "terminal",
        "browser_basic",
        "browser_interaction",
        "research",
        "security",
        "time_context",
    ],
    # Head of Growth — data, experiments, competitor research; no terminal/security.
    "queen_growth": [
        "file_ops",
        "browser_basic",
        "browser_interaction",
        "research",
        "time_context",
    ],
    # Head of Product Strategy — user research + roadmaps; no terminal/security.
    "queen_product_strategy": [
        "file_ops",
        "browser_basic",
        "browser_interaction",
        "research",
        "time_context",
    ],
    # Head of Finance — financial models (CSV/Excel heavy), market research.
    "queen_finance_fundraising": [
        "file_ops",
        "advanced_spreadsheet",
        "browser_basic",
        "browser_interaction",
        "research",
        "time_context",
    ],
    # Head of Legal — reads contracts/PDFs, researches; no terminal/data/security.
    "queen_legal": [
        "file_ops",
        "browser_basic",
        "browser_interaction",
        "research",
        "time_context",
    ],
    # Head of Brand & Design — visual refs, style guides; no terminal/data/security.
    "queen_brand_design": [
        "file_ops",
        "browser_basic",
        "browser_interaction",
        "research",
        "time_context",
    ],
    # Head of Talent — candidate pipelines, resumes; data + browser heavy.
    "queen_talent": [
        "file_ops",
        "browser_basic",
        "browser_interaction",
        "research",
        "time_context",
    ],
    # Head of Operations — processes, automation, observability.
    "queen_operations": [
        "file_ops",
        "data",
        "browser_basic",
        "browser_interaction",
        "research",
        "time_context",
    ],
}


def has_role_default(queen_id: str) -> bool:
    """Return True when ``queen_id`` is known to the category table."""
    return queen_id in QUEEN_DEFAULT_CATEGORIES


def resolve_queen_default_tools(
    queen_id: str,
    mcp_catalog: dict[str, list[dict[str, Any]]] | None = None,
) -> list[str] | None:
    """Return the role-based default allowlist for ``queen_id``.

    Arguments:
        queen_id: Profile ID (e.g. ``"queen_technology"``).
        mcp_catalog: Optional mapping of ``{server_name: [{"name": ...}, ...]}``
            used to expand ``@server:NAME`` shorthands in categories.
            When absent, shorthand entries are dropped and the result
            contains only the explicitly-named tools.

    Returns:
        A deduplicated list of tool names, or ``None`` if the queen has
        no role entry (caller should treat as "allow every MCP tool").
    """
    categories = QUEEN_DEFAULT_CATEGORIES.get(queen_id)
    if not categories:
        return None

    names: list[str] = []
    seen: set[str] = set()

    def _add(name: str) -> None:
        if name and name not in seen:
            seen.add(name)
            names.append(name)

    for cat in categories:
        for entry in _TOOL_CATEGORIES.get(cat, []):
            if entry.startswith("@server:"):
                server_name = entry[len("@server:") :]
                if mcp_catalog is None:
                    logger.debug(
                        "resolve_queen_default_tools: catalog missing; cannot expand %s",
                        entry,
                    )
                    continue
                for tool in mcp_catalog.get(server_name, []) or []:
                    tname = tool.get("name") if isinstance(tool, dict) else None
                    if tname:
                        _add(tname)
            else:
                _add(entry)

    return names
