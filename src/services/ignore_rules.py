"""
Engine-agnostic gitignore-aware path filtering for ingestion (feature 21).

cgr walks the repo with its own fixed ignore list and never consults
``.gitignore`` — build output (``.next/``, ``dist/``, ...), minified bundles,
and anything else the repo's own gitignore excludes gets ingested, embedded,
and (worse) sent through the Tier-2 describe pass. :func:`filter_graph` is
applied to the raw cgr graph JSON, right after cgr runs and before chunking,
so every downstream consumer (chunker, graph_store, the UI) only ever sees a
clean graph.

Two filtering layers, always both applied:
  1. ``git check-ignore --stdin -z`` (git repos only) — authoritative,
     respects nested/global gitignore. ONE batched subprocess for the whole
     path list, never one call per path.
  2. Curated defaults (dir segments + basename globs) — catches junk that is
     often TRACKED (so git wouldn't ignore it) or that a future non-git/
     native engine might otherwise re-ingest.

A non-git repo (or any git failure) falls back to defaults-only — see
:func:`_git_check_ignore`.
"""

from __future__ import annotations

import fnmatch
import logging
import os
import subprocess
from typing import Any

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 30

# Directory segments considered junk regardless of git status. Segments cgr
# already skips on its own are harmless to re-list here — cheap insurance
# for a future native (non-cgr) ingestion engine. Case-sensitive, checked as
# exact path segments (not substrings).
DEFAULT_IGNORED_DIR_SEGMENTS = frozenset(
    {
        ".next",
        ".nuxt",
        ".output",
        ".svelte-kit",
        ".turbo",
        "storybook-static",
        "dist",
        "build",
        "coverage",
    }
)

# Basename glob patterns considered junk regardless of git status.
DEFAULT_IGNORED_BASENAME_GLOBS = frozenset({"*.min.js", "*.min.css", "*.map"})


def _is_default_ignored(rel_path: str) -> bool:
    """True if *rel_path* (repo-relative, ``/``-separated) matches a curated
    default: a directory segment anywhere in the path, or a basename glob."""
    parts = rel_path.split("/")
    if any(part in DEFAULT_IGNORED_DIR_SEGMENTS for part in parts):
        return True
    basename = parts[-1] if parts else rel_path
    return any(fnmatch.fnmatch(basename, pattern) for pattern in DEFAULT_IGNORED_BASENAME_GLOBS)


def _git_check_ignore(repo_root: str, rel_paths: list[str]) -> tuple[set[str], str]:
    """
    Batched ``git check-ignore --stdin -z`` — ONE subprocess for the whole
    *rel_paths* list. Returns ``(excluded, git_status)``.

    ``git_status`` — one of:
      - ``"ok"``: git ran successfully (whether or not anything matched), or
        there was nothing to check.
      - ``"not-a-repo"``: git's own "not a git repository" error — the
        expected, harmless case for a non-git *repo_root*.
      - ``"error"``: any OTHER failure (timeout, git not on PATH, or a real
        git repo git refuses to touch — e.g. the `--full` container's bind
        mount owned by the host user vs. an unprivileged container user
        triggers git's "dubious ownership" guard with exit 128). This is a
        SILENT degradation if not surfaced: gitignore rules are skipped but
        it looks identical to "nothing was gitignored" — callers should
        report this status so it's visible in the job result instead of
        indistinguishable from a clean repo.

    Exit codes: 0 = some paths ignored, 1 = none ignored, 128 = error. Any
    non-{0,1} outcome is fail-open: logged as a warning and treated as
    "nothing ignored by git" (the curated defaults in :func:`classify_paths`
    still apply on top) — only ``git_status`` distinguishes why.
    """
    if not rel_paths:
        return set(), "ok"

    stdin_payload = "\0".join(rel_paths) + "\0"
    try:
        result = subprocess.run(
            ["git", "check-ignore", "--stdin", "-z"],
            cwd=repo_root,
            input=stdin_payload,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.warning(
            "git check-ignore failed in '%s' (%s) — treating repo as non-git "
            "for this run (curated defaults still apply)",
            repo_root,
            e,
        )
        return set(), "error"

    if result.returncode not in (0, 1):
        stderr = result.stderr.strip()
        git_status = "not-a-repo" if "not a git repository" in stderr else "error"
        logger.warning(
            "git check-ignore exited %d in '%s' (stderr: %s) — treating repo "
            "as non-git for this run (curated defaults still apply); "
            "git_status=%s",
            result.returncode,
            repo_root,
            stderr,
            git_status,
        )
        return set(), git_status

    if not result.stdout:
        return set(), "ok"
    return {p for p in result.stdout.split("\0") if p}, "ok"


def classify_paths(repo_root: str, rel_paths: list[str]) -> tuple[set[str], dict[str, Any]]:
    """
    Classify *rel_paths* (repo-relative, ``/``-separated, deduplicated by the
    caller) as excluded or kept.

    Returns ``(excluded, stats)`` where ``stats`` has
    ``files_filtered_gitignore``, ``files_filtered_defaults`` (each counting
    distinct paths, not graph nodes) and ``git_status`` (see
    :func:`_git_check_ignore`). Curated defaults are always applied; ``git
    check-ignore`` is only run against whatever remains after the defaults
    (no point asking git about a path already excluded).
    """
    stats: dict[str, Any] = {
        "files_filtered_gitignore": 0,
        "files_filtered_defaults": 0,
        "git_status": "ok",
    }
    if not rel_paths:
        return set(), stats

    default_excluded = {p for p in rel_paths if _is_default_ignored(p)}
    stats["files_filtered_defaults"] = len(default_excluded)

    remaining = [p for p in rel_paths if p not in default_excluded]
    git_excluded, git_status = _git_check_ignore(repo_root, remaining)
    stats["files_filtered_gitignore"] = len(git_excluded)
    stats["git_status"] = git_status

    return default_excluded | git_excluded, stats


def _rel_path_for_node(node: dict[str, Any], repo_root_real: str) -> tuple[str | None, bool]:
    """
    Resolve a graph node's repo-relative path for classification.

    Returns ``(rel_path, is_outside_repo)``:
      - ``(None, False)`` — the node carries no ``path`` property (Project,
        ExternalPackage) or resolves to the repo root itself. Never filtered.
      - ``(None, True)`` — the recorded path resolves outside *repo_root_real*
        (an absolute/weird path in the graph) or can't be resolved at all.
        Left alone, just counted — never crashes the filter.
      - ``(rel_path, False)`` — normal case, ``/``-separated and relative to
        the repo root.
    """
    props = node.get("properties", {})
    path = props.get("path")
    if not path:
        return None, False

    candidate = path if os.path.isabs(path) else os.path.join(repo_root_real, path)
    try:
        real = os.path.realpath(candidate)
        rel = os.path.relpath(real, repo_root_real)
    except (OSError, ValueError):
        return None, True

    rel = rel.replace(os.sep, "/")
    if rel == ".":
        return None, False
    if rel == ".." or rel.startswith("../"):
        return None, True
    return rel, False


def filter_graph(data: dict[str, Any], repo_root: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Filter a cgr graph JSON (``{"nodes", "relationships", "metadata"}``),
    removing every node whose repo-relative path is gitignore/default-
    excluded, and every relationship referencing a removed node.

    Node path resolution: every node carrying a ``properties.path`` (Folder,
    Package, File, Module, and every code-entity label — entities are
    filtered by their *file's* path, since that's what ``path`` holds for
    them) is a candidate; ``Project``/``ExternalPackage`` nodes carry no
    ``path`` and are never filtered. The repo root itself is never filtered.
    Paths resolving outside *repo_root* are left alone and only counted
    (``nodes_outside_repo``) — defensive, never raises on an absolute or
    unusual path recorded in the graph.

    ``filtered_data["metadata"]`` has its ``total_nodes``/``total_relationships``
    rewritten to the POST-filter counts (everything else in ``metadata``, e.g.
    ``exported_at``, passes through unchanged) — the canonical artifact must
    not ship metadata that contradicts its own ``nodes``/``relationships``.

    Returns ``(filtered_data, stats)``. ``stats`` — ``files_filtered_gitignore``,
    ``files_filtered_defaults``, ``nodes_removed``, ``edges_removed``,
    ``nodes_outside_repo``, ``git_status`` (see :func:`_git_check_ignore` —
    surfaces a silent gitignore-skip, e.g. git's "dubious ownership" guard in
    a container, distinctly from a genuinely clean/non-git repo).
    """
    repo_root_real = os.path.realpath(repo_root)
    nodes: list[dict[str, Any]] = data.get("nodes", [])
    relationships: list[dict[str, Any]] = data.get("relationships", [])

    node_rel_path: dict[int, str] = {}
    nodes_outside_repo = 0
    for node in nodes:
        rel, outside = _rel_path_for_node(node, repo_root_real)
        if outside:
            nodes_outside_repo += 1
        elif rel is not None:
            node_rel_path[node["node_id"]] = rel

    unique_paths = sorted(set(node_rel_path.values()))
    excluded_paths, class_stats = classify_paths(repo_root_real, unique_paths)

    removed_ids = {nid for nid, rel in node_rel_path.items() if rel in excluded_paths}

    kept_nodes = [n for n in nodes if n["node_id"] not in removed_ids]
    kept_relationships = [
        r
        for r in relationships
        if r["from_id"] not in removed_ids and r["to_id"] not in removed_ids
    ]

    filtered_data = dict(data)
    filtered_data["nodes"] = kept_nodes
    filtered_data["relationships"] = kept_relationships
    metadata = dict(data.get("metadata", {}))
    metadata["total_nodes"] = len(kept_nodes)
    metadata["total_relationships"] = len(kept_relationships)
    filtered_data["metadata"] = metadata

    stats = {
        "files_filtered_gitignore": class_stats["files_filtered_gitignore"],
        "files_filtered_defaults": class_stats["files_filtered_defaults"],
        "nodes_removed": len(removed_ids),
        "edges_removed": len(relationships) - len(kept_relationships),
        "nodes_outside_repo": nodes_outside_repo,
        "git_status": class_stats["git_status"],
    }
    if stats["nodes_removed"] or stats["edges_removed"]:
        logger.info(
            "ignore_rules.filter_graph: removed %d node(s) / %d edge(s) from "
            "'%s' (%d gitignored, %d default-excluded path(s))",
            stats["nodes_removed"],
            stats["edges_removed"],
            repo_root,
            stats["files_filtered_gitignore"],
            stats["files_filtered_defaults"],
        )
    return filtered_data, stats
