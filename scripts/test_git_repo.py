#!/usr/bin/env python
"""
Lightweight tests for ``services.git_repo`` — this repo has no test
framework yet, so this is a plain assert-and-print script rather than a new
pytest dependency.

Covers the unified-diff parser against embedded fixture diffs (simple
modify, new file, deleted file, rename+edit, binary, multiple hunks,
no-newline marker), plus a live READ-ONLY smoke test against the sample
corpus repo (skipped gracefully if that path doesn't exist on this machine).

Usage
-----
    uv run python scripts/test_git_repo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the `services` package importable (the app roots absolute imports at src/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from services import git_repo  # noqa: E402

SAMPLE_REPO = "/Users/username/projects/sample_back-end"

_passed = 0


def check(label: str, condition: bool) -> None:
    global _passed
    if not condition:
        raise AssertionError(f"FAIL: {label}")
    _passed += 1
    print(f"PASS: {label}")


# ---------------------------------------------------------------------------
# Fixture diffs
# ---------------------------------------------------------------------------

SIMPLE_MODIFY = """\
diff --git a/foo.py b/foo.py
index e69de29..4b825dc 100644
--- a/foo.py
+++ b/foo.py
@@ -1,3 +1,4 @@
 line1
-line2
+line2 modified
 line3
+line4
"""

NEW_FILE = """\
diff --git a/new_file.py b/new_file.py
new file mode 100644
index 0000000..1234567
--- /dev/null
+++ b/new_file.py
@@ -0,0 +1,2 @@
+line1
+line2
"""

DELETED_FILE = """\
diff --git a/old_file.py b/old_file.py
deleted file mode 100644
index 1234567..0000000
--- a/old_file.py
+++ /dev/null
@@ -1,2 +0,0 @@
-line1
-line2
"""

RENAME_AND_EDIT = """\
diff --git a/old_name.py b/new_name.py
similarity index 92%
rename from old_name.py
rename to new_name.py
index 1234567..89abcdef 100644
--- a/old_name.py
+++ b/new_name.py
@@ -1,3 +1,3 @@
 line1
-line2
+line2 edited
 line3
"""

BINARY_FILE = """\
diff --git a/image.png b/image.png
index 1234567..89abcdef 100644
Binary files a/image.png and b/image.png differ
"""

MULTIPLE_HUNKS = """\
diff --git a/multi.py b/multi.py
index 1111111..2222222 100644
--- a/multi.py
+++ b/multi.py
@@ -1,2 +1,2 @@
-a
+A
@@ -10,2 +10,3 @@
 b
+c
 d
"""

NO_NEWLINE_MARKER = """\
diff --git a/nonewline.py b/nonewline.py
index 1111111..2222222 100644
--- a/nonewline.py
+++ b/nonewline.py
@@ -1,1 +1,1 @@
-old content
\\ No newline at end of file
+new content
\\ No newline at end of file
"""

# Realistic form of a non-ASCII path once diff() pins `-c core.quotePath=false`
# (raw UTF-8, no octal-quoting) — see test_unicode_path_unquoted below.
UNICODE_PATH_MODIFY = """\
diff --git a/café.ts b/café.ts
index 1111111..2222222 100644
--- a/café.ts
+++ b/café.ts
@@ -1,1 +1,1 @@
-old
+new
"""


def test_simple_modify() -> None:
    diffs = git_repo.parse_unified_diff(SIMPLE_MODIFY)
    check("simple_modify: one file", len(diffs) == 1)
    fd = diffs[0]
    check("simple_modify: path", fd.path == "foo.py")
    check("simple_modify: old_path == path", fd.old_path == fd.path)
    check("simple_modify: flags", not (fd.is_new or fd.is_deleted or fd.is_rename or fd.is_binary))
    check("simple_modify: one hunk", len(fd.hunks) == 1)
    hunk = fd.hunks[0]
    check(
        "simple_modify: hunk header",
        (hunk.old_start, hunk.old_count, hunk.new_start, hunk.new_count) == (1, 3, 1, 4),
    )
    check("simple_modify: hunk text has @@ header", hunk.text.startswith("@@ -1,3 +1,4 @@"))
    check("simple_modify: ranges", fd.changed_line_ranges == [(2, 2), (4, 4)])


def test_new_file() -> None:
    diffs = git_repo.parse_unified_diff(NEW_FILE)
    check("new_file: one file", len(diffs) == 1)
    fd = diffs[0]
    check("new_file: is_new", fd.is_new)
    check("new_file: not deleted/rename/binary", not (fd.is_deleted or fd.is_rename or fd.is_binary))
    check("new_file: old_path == path", fd.old_path == fd.path == "new_file.py")
    check("new_file: ranges", fd.changed_line_ranges == [(1, 2)])


def test_deleted_file() -> None:
    diffs = git_repo.parse_unified_diff(DELETED_FILE)
    check("deleted_file: one file", len(diffs) == 1)
    fd = diffs[0]
    check("deleted_file: is_deleted", fd.is_deleted)
    check("deleted_file: old_path == path", fd.old_path == fd.path == "old_file.py")
    check("deleted_file: pure-deletion single point", fd.changed_line_ranges == [(0, 0)])


def test_rename_and_edit() -> None:
    diffs = git_repo.parse_unified_diff(RENAME_AND_EDIT)
    check("rename_and_edit: one file", len(diffs) == 1)
    fd = diffs[0]
    check("rename_and_edit: is_rename", fd.is_rename)
    check("rename_and_edit: old_path", fd.old_path == "old_name.py")
    check("rename_and_edit: path", fd.path == "new_name.py")
    check("rename_and_edit: old_path != path", fd.old_path != fd.path)
    check("rename_and_edit: ranges", fd.changed_line_ranges == [(2, 2)])


def test_binary_file() -> None:
    diffs = git_repo.parse_unified_diff(BINARY_FILE)
    check("binary_file: one file", len(diffs) == 1)
    fd = diffs[0]
    check("binary_file: is_binary", fd.is_binary)
    check("binary_file: no hunks", fd.hunks == [])
    check("binary_file: no ranges", fd.changed_line_ranges == [])


def test_multiple_hunks() -> None:
    diffs = git_repo.parse_unified_diff(MULTIPLE_HUNKS)
    check("multiple_hunks: one file", len(diffs) == 1)
    fd = diffs[0]
    check("multiple_hunks: two hunks", len(fd.hunks) == 2)
    check("multiple_hunks: ranges", fd.changed_line_ranges == [(1, 1), (11, 11)])


def test_no_newline_marker() -> None:
    diffs = git_repo.parse_unified_diff(NO_NEWLINE_MARKER)
    check("no_newline_marker: one file", len(diffs) == 1)
    fd = diffs[0]
    check("no_newline_marker: one hunk", len(fd.hunks) == 1)
    check("no_newline_marker: marker preserved in text", "No newline at end of file" in fd.hunks[0].text)
    check("no_newline_marker: ranges (mixed -/+ merge)", fd.changed_line_ranges == [(1, 1)])


def test_unicode_path_unquoted() -> None:
    # Confirms the "realistic case" the -c core.quotePath=false invocation
    # flag produces (raw UTF-8 path, no octal-quoting) parses cleanly —
    # genuinely-control-character paths (still quoted even with that flag)
    # remain a known, accepted-by-review limitation.
    diffs = git_repo.parse_unified_diff(UNICODE_PATH_MODIFY)
    check("unicode_path: one file", len(diffs) == 1)
    fd = diffs[0]
    check("unicode_path: path", fd.path == "café.ts")
    check("unicode_path: old_path == path", fd.old_path == "café.ts")


def test_diff_argv_pins_prefixes_quotepath_and_renames() -> None:
    """
    diff() must never trust the caller's ambient git config: this asserts
    the constructed argv pins --src-prefix/--dst-prefix (overrides
    diff.noprefix), -c core.quotePath=false (raw UTF-8 paths, no octal
    quoting), and --find-renames — without actually invoking git.
    """
    captured: dict = {}

    class _FakeResult:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _FakeResult()

    real_subprocess_run = git_repo.subprocess.run
    real_merge_base = git_repo.merge_base
    git_repo.subprocess.run = fake_run
    git_repo.merge_base = lambda repo_path, base, target: "0" * 40
    try:
        git_repo.diff("/fake/repo", "main", "feature")
    finally:
        git_repo.subprocess.run = real_subprocess_run
        git_repo.merge_base = real_merge_base

    cmd = captured["cmd"]
    check("diff argv: -c core.quotePath=false", "-c" in cmd and "core.quotePath=false" in cmd)
    check("diff argv: --src-prefix=a/", "--src-prefix=a/" in cmd)
    check("diff argv: --dst-prefix=b/", "--dst-prefix=b/" in cmd)
    check("diff argv: --find-renames", "--find-renames" in cmd)


def test_diff_stat_argv_pins_renames() -> None:
    """diff_stat()'s --find-renames must match diff()'s (same invariant)."""
    captured: dict = {}

    class _FakeResult:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _FakeResult()

    real_subprocess_run = git_repo.subprocess.run
    real_merge_base = git_repo.merge_base
    git_repo.subprocess.run = fake_run
    git_repo.merge_base = lambda repo_path, base, target: "0" * 40
    try:
        git_repo.diff_stat("/fake/repo", "main", "feature")
    finally:
        git_repo.subprocess.run = real_subprocess_run
        git_repo.merge_base = real_merge_base

    cmd = captured["cmd"]
    check("diff_stat argv: --find-renames", "--find-renames" in cmd)


def test_multi_file_diff() -> None:
    combined = SIMPLE_MODIFY + NEW_FILE + BINARY_FILE
    diffs = git_repo.parse_unified_diff(combined)
    check("multi_file_diff: three files", len(diffs) == 3)
    check(
        "multi_file_diff: order preserved",
        [fd.path for fd in diffs] == ["foo.py", "new_file.py", "image.png"],
    )


def test_live_smoke() -> None:
    if not Path(SAMPLE_REPO).is_dir():
        print(f"SKIP: live sample smoke test — '{SAMPLE_REPO}' not found on this machine")
        return

    check("live: is_git_repo", git_repo.is_git_repo(SAMPLE_REPO))

    branches = git_repo.list_branches(SAMPLE_REPO)
    check("live: has branches", len(branches) > 0)
    current = [b for b in branches if b["is_current"]]
    check("live: exactly one current branch", len(current) == 1)
    check("live: current branch listed first", branches[0]["is_current"])

    base = git_repo.default_base(SAMPLE_REPO)
    print(f"  live: default_base = {base!r}")
    check("live: default_base resolves to a branch name", isinstance(base, str) and base)

    target = "fix/center-to-box"
    branch_names = {b["name"] for b in branches}
    if target not in branch_names:
        print(f"SKIP: live diff smoke test — branch '{target}' not found in {SAMPLE_REPO}")
        return

    mb = git_repo.merge_base(SAMPLE_REPO, base, target)
    print(f"  live: merge_base({base}, {target}) = {mb}")
    check("live: merge_base returns a sha", len(mb) == 40)

    stat = git_repo.diff_stat(SAMPLE_REPO, base, target)
    print(f"  live: diff_stat = {stat}")

    diff_text = git_repo.diff(SAMPLE_REPO, base, target)
    check("live: diff produced output", len(diff_text) > 0)

    file_diffs = git_repo.parse_unified_diff(diff_text)
    total_hunks = sum(len(fd.hunks) for fd in file_diffs)
    total_ranges = sum(len(fd.changed_line_ranges) for fd in file_diffs)
    print(
        f"  live: fix/center-to-box parsed — files={len(file_diffs)} "
        f"hunks={total_hunks} changed_line_ranges={total_ranges}"
    )
    check("live: parsed at least one file", len(file_diffs) > 0)
    check(
        "live: files_changed matches numstat count",
        len(file_diffs) == stat["files_changed"],
    )

    for fd in file_diffs:
        for start, end in fd.changed_line_ranges:
            check(f"live: range sane in {fd.path}", start >= 0 and end >= start)

    sha = git_repo.resolve_ref(SAMPLE_REPO, target)
    check("live: resolve_ref returns a sha", len(sha) == 40)

    try:
        git_repo.resolve_ref(SAMPLE_REPO, "-not-a-real-flag")
        check("live: resolve_ref rejects leading-dash ref", False)
    except git_repo.GitError:
        check("live: resolve_ref rejects leading-dash ref", True)

    try:
        git_repo.resolve_ref(SAMPLE_REPO, "this-branch-does-not-exist-xyz")
        check("live: resolve_ref rejects unknown ref", False)
    except git_repo.GitError:
        check("live: resolve_ref rejects unknown ref", True)


def main() -> None:
    test_simple_modify()
    test_new_file()
    test_deleted_file()
    test_rename_and_edit()
    test_binary_file()
    test_multiple_hunks()
    test_no_newline_marker()
    test_unicode_path_unquoted()
    test_diff_argv_pins_prefixes_quotepath_and_renames()
    test_diff_stat_argv_pins_renames()
    test_multi_file_diff()
    test_live_smoke()
    print(f"\n{_passed} checks PASSED")


if __name__ == "__main__":
    main()
