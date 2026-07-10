"""
Read-only git access for a project's source repository.

Every function here shells out to the local ``git`` binary with an argv
LIST (never ``shell=True``) and a bounded timeout. This module is READ-ONLY
by contract: it never runs ``checkout``, ``fetch``, ``pull``, ``clean``, or
``reset`` — or any other command that mutates the working tree, the index,
or refs. Callers get diffs, branch listings, and file contents as of a
given ref; the repository on disk is never touched.

Also home to :func:`parse_unified_diff`, a pure function that turns
``git diff`` output into structured :class:`FileDiff`/:class:`Hunk` objects.

All functions are synchronous — callers running under FastAPI/asyncio are
expected to run them in a worker thread (e.g. ``anyio.to_thread`` /
``run_in_executor``).
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field

_TIMEOUT_SECONDS = 30


class GitError(RuntimeError):
    """Raised for any git failure — invalid ref, non-repo path, timeout, ..."""


def _validate_ref(ref: str) -> None:
    """
    Argument-injection guard: reject refs that could be mistaken for a git
    option (leading ``-``) or that contain whitespace (never a valid single
    ref, and a hint of accidental multi-argument injection).
    """
    if not ref or ref.startswith("-"):
        raise GitError(f"invalid git ref '{ref}'")
    if any(ch.isspace() for ch in ref):
        raise GitError(f"invalid git ref '{ref}'")


def _run(repo_path: str, args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run ``git <args>`` in *repo_path*, capturing output as text."""
    try:
        return subprocess.run(
            ["git", *args],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as e:
        raise GitError(
            f"git {' '.join(args)} timed out after {_TIMEOUT_SECONDS}s in '{repo_path}'"
        ) from e
    except OSError as e:
        raise GitError(f"failed to run git in '{repo_path}': {e}") from e


def is_git_repo(repo_path: str) -> bool:
    """True if *repo_path* is inside a git working tree."""
    result = _run(repo_path, ["rev-parse", "--is-inside-work-tree"])
    return result.returncode == 0 and result.stdout.strip() == "true"


def list_branches(repo_path: str) -> list[dict]:
    """
    List local branches: ``{"name", "is_current", "last_commit_date"}``
    (ISO 8601), sorted current-first then most-recently-committed first.
    """
    result = _run(
        repo_path,
        [
            "for-each-ref",
            "refs/heads",
            "--format=%(refname:short)\t%(HEAD)\t%(committerdate:iso-strict)",
        ],
    )
    if result.returncode != 0:
        raise GitError(
            f"failed to list branches in '{repo_path}': {result.stderr.strip()}"
        )

    branches = []
    for line in result.stdout.splitlines():
        if not line:
            continue
        name, head_marker, commit_date = line.split("\t")
        branches.append(
            {
                "name": name,
                "is_current": head_marker == "*",
                "last_commit_date": commit_date,
            }
        )

    # Current-first, then most-recently-committed first. Two-key sort: sort
    # by date descending first, then a stable sort on "not is_current" keeps
    # that date ordering within each group while moving the current branch
    # to the front.
    branches.sort(key=lambda b: b["last_commit_date"], reverse=True)
    branches.sort(key=lambda b: not b["is_current"])
    return branches


def default_base(repo_path: str) -> str:
    """
    The repository's default base branch: ``origin/HEAD``'s target if set,
    else the first of ``main``/``master`` that exists as a local branch.

    Raises :class:`GitError` (actionable) when neither is resolvable.
    """
    result = _run(repo_path, ["symbolic-ref", "refs/remotes/origin/HEAD"])
    if result.returncode == 0:
        ref = result.stdout.strip()
        prefix = "refs/remotes/origin/"
        if ref.startswith(prefix):
            return ref[len(prefix):]

    for candidate in ("main", "master"):
        check = _run(repo_path, ["rev-parse", "--verify", f"refs/heads/{candidate}"])
        if check.returncode == 0:
            return candidate

    raise GitError(
        f"could not determine a default base branch for repository '{repo_path}' — "
        "no origin/HEAD and no local 'main' or 'master' branch"
    )


def resolve_ref(repo_path: str, ref: str) -> str:
    """Resolve *ref* to its full commit SHA, or raise :class:`GitError`."""
    _validate_ref(ref)
    result = _run(repo_path, ["rev-parse", "--verify", f"{ref}^{{commit}}"])
    if result.returncode != 0:
        raise GitError(f"unknown git ref '{ref}' in repository '{repo_path}'")
    return result.stdout.strip()


def merge_base(repo_path: str, base: str, target: str) -> str:
    """The merge-base commit SHA of *base* and *target*."""
    _validate_ref(base)
    _validate_ref(target)
    result = _run(repo_path, ["merge-base", base, target])
    if result.returncode != 0:
        raise GitError(
            f"failed to compute merge-base of '{base}' and '{target}' "
            f"in repository '{repo_path}': {result.stderr.strip()}"
        )
    return result.stdout.strip()


def diff(repo_path: str, base: str, target: str) -> str:
    """
    Unified diff of *base* to *target*, equivalent to three-dot
    ``base...target`` — computed deterministically as the two-dot diff from
    their explicit merge-base to *target*.

    ``-c core.quotePath=false`` forces raw UTF-8 paths in the output instead
    of octal-quoted ones (never trust the caller's ambient git config for
    this — :func:`parse_unified_diff` doesn't understand quoted paths).
    ``--src-prefix=a/ --dst-prefix=b/`` pins the ``diff --git a/... b/...``
    prefixes the parser hardcodes, overriding a user's ``diff.noprefix``.
    """
    mb = merge_base(repo_path, base, target)
    _validate_ref(target)
    result = _run(
        repo_path,
        [
            "-c",
            "core.quotePath=false",
            "diff",
            "--no-color",
            "--unified=3",
            "--find-renames",
            "--src-prefix=a/",
            "--dst-prefix=b/",
            mb,
            target,
        ],
    )
    if result.returncode != 0:
        raise GitError(
            f"failed to diff '{base}...{target}' in repository '{repo_path}': "
            f"{result.stderr.strip()}"
        )
    return result.stdout


def diff_stat(repo_path: str, base: str, target: str) -> dict:
    """
    ``{"files_changed", "insertions", "deletions"}`` for ``base...target``
    (same merge-base semantics as :func:`diff`), via ``--numstat``.

    ``--find-renames`` matches :func:`diff`'s detection setting — without it,
    a user's ``diff.renames=false`` would make a rename+edit count as two
    files here but one in :func:`parse_unified_diff`'s output.
    """
    mb = merge_base(repo_path, base, target)
    _validate_ref(target)
    result = _run(repo_path, ["diff", "--numstat", "--find-renames", mb, target])
    if result.returncode != 0:
        raise GitError(
            f"failed to diff-stat '{base}...{target}' in repository '{repo_path}': "
            f"{result.stderr.strip()}"
        )

    files_changed = 0
    insertions = 0
    deletions = 0
    for line in result.stdout.splitlines():
        if not line:
            continue
        added, removed, _path = line.split("\t", 2)
        files_changed += 1
        if added != "-":
            insertions += int(added)
        if removed != "-":
            deletions += int(removed)

    return {
        "files_changed": files_changed,
        "insertions": insertions,
        "deletions": deletions,
    }


def show_file(repo_path: str, ref: str, path: str) -> str | None:
    """
    Contents of *path* as of *ref*, or ``None`` if the file doesn't exist at
    that ref. Any other failure raises :class:`GitError`.
    """
    _validate_ref(ref)
    result = _run(repo_path, ["show", f"{ref}:{path}"])
    if result.returncode == 0:
        return result.stdout

    stderr = result.stderr
    if "does not exist" in stderr or "exists on disk, but not" in stderr:
        return None
    raise GitError(
        f"failed to read '{path}' at ref '{ref}' in repository '{repo_path}': "
        f"{stderr.strip()}"
    )


# ---------------------------------------------------------------------------
# Unified-diff parser
# ---------------------------------------------------------------------------

_HUNK_HEADER_RE = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@"
)


@dataclass
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    text: str


@dataclass
class FileDiff:
    path: str
    old_path: str
    is_new: bool = False
    is_deleted: bool = False
    is_rename: bool = False
    is_binary: bool = False
    hunks: list[Hunk] = field(default_factory=list)
    changed_line_ranges: list[tuple[int, int]] = field(default_factory=list)


def _split_paths(diff_git_line: str) -> tuple[str, str]:
    """
    Best-effort split of a ``diff --git a/OLD b/NEW`` line into
    ``(old, new)``. Ambiguous only for paths containing the literal
    substring `` b/`` — the same inherent ambiguity git's own format has;
    ``rename from``/``rename to`` and ``new/deleted file mode`` markers are
    the authoritative source when present.
    """
    body = diff_git_line[len("diff --git a/"):]
    marker = " b/"
    idx = body.rfind(marker)
    if idx == -1:
        return body, body
    return body[:idx], body[idx + len(marker):]


def _hunk_changed_ranges(hunk_lines: list[str], new_start: int) -> list[tuple[int, int]]:
    """
    New-side line ranges touched by a single hunk: contiguous runs of ``+``/
    ``-`` lines. A run containing at least one ``+`` line is reported as the
    span of its ``+`` line positions; a run of pure ``-`` lines is reported
    as the single new-side position where the deletion happened (the
    position doesn't advance while consuming ``-`` lines).
    """
    ranges: list[tuple[int, int]] = []
    new_line_no = new_start

    in_run = False
    run_has_plus = False
    run_start_no = new_line_no
    plus_start: int | None = None
    plus_end: int | None = None

    def flush() -> None:
        nonlocal in_run, run_has_plus, plus_start, plus_end
        if not in_run:
            return
        if run_has_plus:
            ranges.append((plus_start, plus_end))
        else:
            ranges.append((run_start_no, run_start_no))
        in_run = False
        run_has_plus = False
        plus_start = None
        plus_end = None

    for line in hunk_lines:
        if line.startswith("\\"):
            # "\ No newline at end of file" — no line-number effect.
            continue
        if line.startswith("+"):
            if not in_run:
                in_run = True
                run_start_no = new_line_no
            run_has_plus = True
            if plus_start is None:
                plus_start = new_line_no
            plus_end = new_line_no
            new_line_no += 1
        elif line.startswith("-"):
            if not in_run:
                in_run = True
                run_start_no = new_line_no
            # deletions don't advance the new-side counter
        else:
            # context line (starts with a space, or is a stray blank line)
            flush()
            new_line_no += 1

    flush()
    return ranges


def parse_unified_diff(diff_text: str) -> list[FileDiff]:
    """
    Parse ``git diff`` output (as produced by :func:`diff`) into one
    :class:`FileDiff` per file entry, in order.

    Handles renames (via ``similarity index``/``rename from``/``rename
    to``), binary files (``Binary files ... differ``), new/deleted files
    (``/dev/null`` sides), mode-change-only entries (no hunks), CRLF line
    endings within content lines (preserved verbatim — lines are split only
    on ``\\n``), and ``\\ No newline at end of file`` markers.
    """
    if diff_text.endswith("\n"):
        lines = diff_text[:-1].split("\n")
    else:
        lines = diff_text.split("\n")

    file_diffs: list[FileDiff] = []
    i = 0
    n = len(lines)

    while i < n:
        if not lines[i].startswith("diff --git "):
            i += 1
            continue

        old_path, path = _split_paths(lines[i])
        is_new = is_deleted = is_rename = is_binary = False
        hunks: list[Hunk] = []
        changed_line_ranges: list[tuple[int, int]] = []
        i += 1

        while i < n and not lines[i].startswith("diff --git "):
            line = lines[i]

            if line.startswith("rename from "):
                old_path = line[len("rename from "):]
                is_rename = True
                i += 1
            elif line.startswith("rename to "):
                path = line[len("rename to "):]
                is_rename = True
                i += 1
            elif line.startswith("new file mode"):
                is_new = True
                i += 1
            elif line.startswith("deleted file mode"):
                is_deleted = True
                i += 1
            elif line.startswith("Binary files ") and line.endswith(" differ"):
                is_binary = True
                i += 1
            elif line.startswith("@@ "):
                m = _HUNK_HEADER_RE.match(line)
                if m is None:
                    # Malformed/unrecognized header — skip defensively.
                    i += 1
                    continue
                old_start = int(m.group(1))
                old_count = int(m.group(2)) if m.group(2) is not None else 1
                new_start = int(m.group(3))
                new_count = int(m.group(4)) if m.group(4) is not None else 1

                body_start = i
                i += 1
                while i < n and not lines[i].startswith("diff --git ") and not lines[i].startswith("@@ "):
                    i += 1
                hunk_lines = lines[body_start + 1:i]

                hunks.append(
                    Hunk(
                        old_start=old_start,
                        old_count=old_count,
                        new_start=new_start,
                        new_count=new_count,
                        text="\n".join(lines[body_start:i]),
                    )
                )
                changed_line_ranges.extend(_hunk_changed_ranges(hunk_lines, new_start))
            else:
                i += 1

        if is_new:
            old_path = path
        elif is_deleted:
            path = old_path

        file_diffs.append(
            FileDiff(
                path=path,
                old_path=old_path,
                is_new=is_new,
                is_deleted=is_deleted,
                is_rename=is_rename,
                is_binary=is_binary,
                hunks=hunks,
                changed_line_ranges=changed_line_ranges,
            )
        )

    return file_diffs
