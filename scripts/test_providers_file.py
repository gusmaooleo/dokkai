#!/usr/bin/env python
"""
Tests for ``config/providers.json`` loading (feature 22, C3) —
``services.llm_provider``'s ``_load_providers_file()`` and its integration
with ``get_provider()``, ``get_registered_provider_ids()``, and
``key_env_for()``, plus the config layer
(``services.llm_config.validate_and_save_config``) end to end.

The loader runs once, at import time, as a module-level side effect (see
the "File-registered providers" section of ``services/llm_provider.py``) —
so most scenarios here (malformed file, unknown shape, built-in shadowing,
...) each need their OWN fresh interpreter. Every check below drives a
short ``python -c`` snippet in a subprocess (``DOKKAI_PROVIDERS_FILE``
pointed at a scratch fixture, ``src`` on ``PYTHONPATH``) and inspects its
exit code / stdout / stderr — this exercises the REAL production loader
inside ``services.llm_provider``, not a re-implementation of it.

Offline only: no network, no Postgres, no real API key. Every env var
this test resolves ``${...}`` against is a fabricated fixture var, never
one of the real *_API_KEY entries a developer's .env may hold — this
script never loads .env at all, and explicitly strips any *_API_KEY from
the child processes' environment before adding fixture-only vars back in.

Usage
-----
    uv run python scripts/test_providers_file.py
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"

_passed = 0
_failed = 0


def check(label: str, cond: bool, extra: object = "") -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"PASS: {label}")
    else:
        _failed += 1
        print(f"FAIL: {label} — {extra!r}")


def write_fixture(tmpdir: str, name: str, data: object) -> str:
    path = os.path.join(tmpdir, name)
    with open(path, "w") as f:
        if isinstance(data, str):
            f.write(data)
        else:
            json.dump(data, f)
    return path


def run(
    code: str,
    providers_file: str | None,
    extra_env: dict[str, str] | None = None,
    cwd: str | None = None,
) -> subprocess.CompletedProcess:
    """Run *code* in a fresh interpreter with DOKKAI_PROVIDERS_FILE set."""
    env = {
        k: v
        for k, v in os.environ.items()
        # Never let a real .env-loaded *_API_KEY leak into a child whose
        # whole point is testing ${VAR} resolution against fixture vars.
        if not k.endswith("_API_KEY")
    }
    env["PYTHONPATH"] = str(SRC_DIR)
    if providers_file is not None:
        env["DOKKAI_PROVIDERS_FILE"] = providers_file
    else:
        env.pop("DOKKAI_PROVIDERS_FILE", None)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd,
        timeout=30,
    )


BASELINE_IDS = "['anthropic', 'gemini', 'openai']"

IMPORT_AND_PRINT_IDS = (
    "import services.llm_provider as lp; "
    "print(sorted(lp.get_registered_provider_ids()))"
)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        # -----------------------------------------------------------------
        # File absent (explicit override pointing at nothing) — clean
        # no-op, registry identical to the no-file baseline.
        # -----------------------------------------------------------------
        absent_path = os.path.join(tmpdir, "does-not-exist.json")
        r = run(IMPORT_AND_PRINT_IDS, absent_path)
        check(
            "absent file: clean no-op, exits 0",
            r.returncode == 0,
            (r.returncode, r.stdout, r.stderr),
        )
        check(
            "absent file: registry identical to baseline (no ids added)",
            r.stdout.strip() == BASELINE_IDS,
            r.stdout,
        )

        # -----------------------------------------------------------------
        # Default path (no DOKKAI_PROVIDERS_FILE override) resolves
        # relative to cwd. Hermetic: run in a throwaway tmp dir that has no
        # config/ subdirectory of its own, rather than asserting on the
        # REPO's cwd (which breaks the moment a developer follows this
        # feature's own instructions and creates config/providers.json).
        # -----------------------------------------------------------------
        cwd_no_config = os.path.join(tmpdir, "cwd_no_config")
        os.makedirs(cwd_no_config)
        r = run(IMPORT_AND_PRINT_IDS, None, cwd=cwd_no_config)
        check(
            "default path, cwd with no config/ dir: no-op",
            r.returncode == 0 and r.stdout.strip() == BASELINE_IDS,
            (r.returncode, r.stdout, r.stderr),
        )

        # And the positive case: default path DOES resolve and load when
        # cwd/config/providers.json genuinely exists — this is the real
        # "default path works" proof (the old version of this test only
        # proved absence, never presence).
        cwd_with_config = os.path.join(tmpdir, "cwd_with_config")
        os.makedirs(os.path.join(cwd_with_config, "config"))
        with open(os.path.join(cwd_with_config, "config", "providers.json"), "w") as f:
            json.dump(
                {"providers": {"defaultpathid": {"api": "openai-completions", "baseUrl": "http://localhost:9/v1"}}},
                f,
            )
        r = run(IMPORT_AND_PRINT_IDS, None, cwd=cwd_with_config)
        check(
            "default path, cwd/config/providers.json present: loads it",
            r.returncode == 0 and "defaultpathid" in r.stdout,
            (r.returncode, r.stdout, r.stderr),
        )

        # -----------------------------------------------------------------
        # THE REAL DEPLOYMENT CHANNEL: DOKKAI_PROVIDERS_FILE set in a
        # .env file, not the subprocess environment directly (the channel
        # `run()`'s ``providers_file`` param uses everywhere else in this
        # suite, via `extra_env`/the process environment). A .env-only
        # var is the realistic way an admin sets this, and it's the one
        # channel that previously went untested — it exposed a real bug:
        # `services.llm_provider._load_providers_file()` reads
        # `DOKKAI_PROVIDERS_FILE` EAGERLY, as a module-level side effect
        # at import time, and `src/main.py` used to import `controllers`
        # (-> `services.llm_config` -> `services.llm_provider`, triggering
        # that read) BEFORE calling `load_dotenv()` — so a
        # `DOKKAI_PROVIDERS_FILE` living only in `.env` was silently
        # invisible, and a stale `config/providers.json` at the default
        # path (if one existed) was loaded instead, with no error at all.
        #
        # This imports the REAL `main` module — not a hand-typed mirror
        # of its statement order, which could silently drift the moment
        # someone reorders main.py's imports again — with `src/` on
        # PYTHONPATH and cwd pointed at a scratch directory holding a
        # `.env` override AND a STALE default `config/providers.json`,
        # proving main.py's CURRENT import order resolves the .env
        # override (not the stale default, and not nothing) before
        # `services.llm_provider` is first imported.
        #
        # Hermetic: run() strips DOKKAI_PROVIDERS_FILE from the launch
        # environment (providers_file=None) so the ONLY source of the
        # override is the .env file itself; the scratch cwd is outside
        # the real repo tree, so python-dotenv's cwd-based search (the
        # fallback path it takes when invoked via `python -c`, which has
        # no `__main__.__file__`) can never reach the developer's real
        # `.env` at the repo root.
        # -----------------------------------------------------------------
        dotenv_channel_dir = os.path.join(tmpdir, "dotenv_channel")
        os.makedirs(os.path.join(dotenv_channel_dir, "config"))
        os.makedirs(os.path.join(dotenv_channel_dir, "altdir"))
        with open(os.path.join(dotenv_channel_dir, "config", "providers.json"), "w") as f:
            json.dump(
                {"providers": {"staleprovider": {"api": "openai-completions", "baseUrl": "http://stale.example/v1"}}},
                f,
            )
        with open(os.path.join(dotenv_channel_dir, "altdir", "my-providers.json"), "w") as f:
            json.dump(
                {"providers": {"realprovider": {"api": "openai-completions", "baseUrl": "http://real.example/v1"}}},
                f,
            )
        with open(os.path.join(dotenv_channel_dir, ".env"), "w") as f:
            f.write("DOKKAI_PROVIDERS_FILE=altdir/my-providers.json\n")

        main_module_probe = (
            "import main\n"
            "import services.llm_provider as lp\n"
            "print(sorted(lp.get_registered_provider_ids()))\n"
        )
        r = run(main_module_probe, None, cwd=dotenv_channel_dir)
        check(
            "main.py's real import order: importing the actual `main` module succeeds",
            r.returncode == 0,
            (r.returncode, r.stdout, r.stderr),
        )
        check(
            "main.py's real import order: a .env-only DOKKAI_PROVIDERS_FILE is honored "
            "— the file it points at IS loaded ('realprovider' registered)",
            "realprovider" in r.stdout,
            (r.stdout, r.stderr),
        )
        check(
            "main.py's real import order: the STALE default config/providers.json is "
            "NOT loaded instead ('staleprovider' absent)",
            "staleprovider" not in r.stdout,
            r.stdout,
        )

        # -----------------------------------------------------------------
        # verify_providers_file_consistency() (feature 22, C9 review round
        # 2) — belt-and-braces for the exact bug just fixed above: if a
        # FUTURE entry point reintroduces the wrong import order (imports
        # this module before its own load_dotenv() runs), this must fail
        # LOUDLY at boot instead of silently serving the wrong file again.
        # Exercised directly (not through `main`, which no longer
        # reproduces the broken order at all now that it's fixed) —
        # reusing the SAME fixture directory as the block above (`.env`
        # naming altdir/my-providers.json as the override, a STALE
        # config/providers.json at the default path).
        # -----------------------------------------------------------------
        consistency_fixed_probe = (
            "from dotenv import load_dotenv\n"
            "load_dotenv()\n"
            "import services.llm_provider as lp\n"
            "lp.verify_providers_file_consistency()\n"
            "print('no error')\n"
        )
        r = run(consistency_fixed_probe, None, cwd=dotenv_channel_dir)
        check(
            "consistency check: load_dotenv() BEFORE import (the fixed main.py order) stays silent",
            r.returncode == 0 and "no error" in r.stdout,
            (r.returncode, r.stdout, r.stderr),
        )

        consistency_broken_probe = (
            "import services.llm_provider as lp\n"  # import BEFORE load_dotenv() — the OLD bug
            "from dotenv import load_dotenv\n"
            "load_dotenv()\n"
            "lp.verify_providers_file_consistency()\n"
            "print('no error')\n"
        )
        r = run(consistency_broken_probe, None, cwd=dotenv_channel_dir)
        check(
            "consistency check: import BEFORE load_dotenv() (the old bug, reintroduced) raises",
            r.returncode != 0 and "inconsistency" in r.stderr,
            (r.returncode, r.stdout, r.stderr),
        )
        check(
            "...and the error names BOTH the actually-loaded (stale, default) path and the "
            "now-expected (.env-provided) path",
            "config/providers.json" in r.stderr and "altdir" in r.stderr and "my-providers.json" in r.stderr,
            r.stderr,
        )

        no_override_dir = os.path.join(tmpdir, "no_override_at_all")
        os.makedirs(no_override_dir)
        r = run(consistency_fixed_probe, None, cwd=no_override_dir)
        check(
            "consistency check: no override at all (no .env, no default file present) stays silent",
            r.returncode == 0 and "no error" in r.stdout,
            (r.returncode, r.stdout, r.stderr),
        )
        r = run(consistency_broken_probe, None, cwd=no_override_dir)
        check(
            "consistency check: no override at all, even under the broken import order, stays silent "
            "(nothing was loaded and nothing should have been)",
            r.returncode == 0 and "no error" in r.stdout,
            (r.returncode, r.stdout, r.stderr),
        )

        env_preset_probe = (
            "import services.llm_provider as lp\n"  # import BEFORE load_dotenv() — broken order
            "from dotenv import load_dotenv\n"
            "load_dotenv()\n"  # no .env file to find here — harmless no-op
            "lp.verify_providers_file_consistency()\n"
            "print('no error')\n"
        )
        real_file_abs_path = os.path.join(dotenv_channel_dir, "altdir", "my-providers.json")
        r = run(env_preset_probe, real_file_abs_path, cwd=no_override_dir)
        check(
            "consistency check: DOKKAI_PROVIDERS_FILE already set in the PROCESS environment "
            "(not only via .env) stays silent even under the broken import order — the var was "
            "already visible at import time, so there was never a mismatch to begin with",
            r.returncode == 0 and "no error" in r.stdout,
            (r.returncode, r.stdout, r.stderr),
        )

        # -----------------------------------------------------------------
        # Empty registration (valid JSON, no providers) — no-op.
        # -----------------------------------------------------------------
        empty_path = write_fixture(tmpdir, "empty.json", {"providers": {}})
        r = run(IMPORT_AND_PRINT_IDS, empty_path)
        check(
            "empty providers object: no-op, registry unchanged",
            r.returncode == 0 and r.stdout.strip() == BASELINE_IDS,
            (r.returncode, r.stdout, r.stderr),
        )

        # -----------------------------------------------------------------
        # A missing 'providers' key — including via a top-level typo like
        # "Providers"/"provider" — must be LOUD, not a silent no-op: that
        # would look exactly like a working system registering nothing,
        # the same trap malformed JSON avoids. Unrelated extra top-level
        # keys (e.g. a "$schema" hint) are tolerated.
        # -----------------------------------------------------------------
        for bad_top_level in ({}, {"Providers": {"x": {}}}, {"provider": {"x": {}}}):
            missing_key_path = write_fixture(tmpdir, f"missing_key_{id(bad_top_level)}.json", bad_top_level)
            r = run(IMPORT_AND_PRINT_IDS, missing_key_path)
            check(
                f"missing/mis-cased top-level 'providers' key is loud, not a no-op: {bad_top_level!r}",
                r.returncode != 0 and "providers" in r.stderr,
                (r.returncode, r.stdout, r.stderr),
            )

        schema_hint_path = write_fixture(
            tmpdir, "schema_hint.json", {"$schema": "https://example.com/providers.schema.json", "providers": {}}
        )
        r = run(IMPORT_AND_PRINT_IDS, schema_hint_path)
        check(
            "an unrelated extra top-level key ('$schema') is tolerated",
            r.returncode == 0 and r.stdout.strip() == BASELINE_IDS,
            (r.returncode, r.stdout, r.stderr),
        )

        # -----------------------------------------------------------------
        # Malformed JSON — loud error at boot naming the path and parse
        # position, not a silent "no providers registered".
        # -----------------------------------------------------------------
        malformed_path = write_fixture(tmpdir, "malformed.json", "{not valid json")
        r = run(IMPORT_AND_PRINT_IDS, malformed_path)
        check(
            "malformed JSON: import fails (nonzero exit)",
            r.returncode != 0,
            (r.returncode, r.stdout, r.stderr),
        )
        check(
            "malformed JSON: error names the file path",
            malformed_path in r.stderr,
            r.stderr,
        )
        check(
            "malformed JSON: error names line/column parse position",
            "line" in r.stderr and "column" in r.stderr,
            r.stderr,
        )

        # -----------------------------------------------------------------
        # Unknown api shape — loud error naming the valid shapes, never a
        # silent fallback to openai-completions.
        # -----------------------------------------------------------------
        bad_shape_path = write_fixture(
            tmpdir,
            "bad_shape.json",
            {"providers": {"foo": {"api": "cohere-native", "baseUrl": "https://x.example/v1"}}},
        )
        r = run(IMPORT_AND_PRINT_IDS, bad_shape_path)
        check(
            "unknown api shape: import fails",
            r.returncode != 0,
            (r.returncode, r.stdout, r.stderr),
        )
        check(
            "unknown api shape: error names the bad value",
            "cohere-native" in r.stderr,
            r.stderr,
        )
        check(
            "unknown api shape: error names both valid shapes",
            "openai-completions" in r.stderr and "anthropic-messages" in r.stderr,
            r.stderr,
        )

        # -----------------------------------------------------------------
        # Built-in shadowing — rejected by name, for all four reserved ids.
        # -----------------------------------------------------------------
        for reserved in ("openai", "gemini", "anthropic", "ollama"):
            shadow_path = write_fixture(
                tmpdir,
                f"shadow_{reserved}.json",
                {
                    "providers": {
                        reserved: {
                            "api": "openai-completions",
                            "baseUrl": "https://evil.example/v1",
                        }
                    }
                },
            )
            r = run(IMPORT_AND_PRINT_IDS, shadow_path)
            check(
                f"built-in shadow rejected: '{reserved}'",
                r.returncode != 0 and "shadows a built-in" in r.stderr,
                (r.returncode, r.stderr),
            )

        # Case-insensitive shadow check too (a typo-adjacent trap: 'OpenAI').
        shadow_case_path = write_fixture(
            tmpdir,
            "shadow_case.json",
            {"providers": {"OpenAI": {"api": "openai-completions", "baseUrl": "https://evil.example/v1"}}},
        )
        r = run(IMPORT_AND_PRINT_IDS, shadow_case_path)
        check(
            "built-in shadow rejected case-insensitively: 'OpenAI'",
            r.returncode != 0 and "shadows a built-in" in r.stderr,
            (r.returncode, r.stderr),
        )

        # -----------------------------------------------------------------
        # Empty/whitespace-only id — rejected, not silently registered
        # (which would otherwise poison "Registered providers: , anthropic, ...").
        # -----------------------------------------------------------------
        for bad_id in ("", "   "):
            blank_id_path = write_fixture(
                tmpdir,
                f"blank_id_{len(bad_id)}.json",
                {"providers": {bad_id: {"api": "openai-completions", "baseUrl": "https://x.example/v1"}}},
            )
            r = run(IMPORT_AND_PRINT_IDS, blank_id_path)
            check(
                f"empty/whitespace-only id rejected: {bad_id!r}",
                r.returncode != 0 and "empty" in r.stderr,
                (r.returncode, r.stderr),
            )

        # -----------------------------------------------------------------
        # Id charset (feature 22, C5 review, round 2): [a-z0-9._-] after
        # normalization — underscore IS included ('lm_studio' is a
        # plausible real id, e.g. for LM Studio, and shouldn't be lost to
        # defend against a frontend-only sentinel string). A space
        # ('corp gateway', which loaded cleanly before this rule existed)
        # and other punctuation outside the set are still rejected.
        # -----------------------------------------------------------------
        for bad_id in ("corp gateway", "gr#oq", "gr/oq", "gr@oq"):
            bad_charset_path = write_fixture(
                tmpdir,
                f"bad_charset_{abs(hash(bad_id))}.json",
                {"providers": {bad_id: {"api": "openai-completions", "baseUrl": "https://x.example/v1"}}},
            )
            r = run(IMPORT_AND_PRINT_IDS, bad_charset_path)
            check(
                f"invalid-charset id rejected: {bad_id!r}",
                r.returncode != 0 and "invalid" in r.stderr and bad_id in r.stderr,
                (r.returncode, r.stderr),
            )

        # -----------------------------------------------------------------
        # Id-rejection messages redact embedded userinfo (feature 22, C5
        # review, round 3): the realistic trigger for THIS specific rule
        # is an admin pasting a gateway URL into the id SLOT instead of
        # baseUrl — the resulting boot-time exception must not echo the
        # credential to stderr (which ships to uvicorn logs / `docker
        # compose logs`, readable by more people than the gitignored
        # file). Only the invalid-charset message is independently
        # exercisable with a credential-bearing id: the shadow/duplicate
        # messages downstream are reached only AFTER an id has already
        # passed the charset gate, and a charset-legal id (no ':', '/',
        # '@') can never itself contain a "://user:pass@" pattern for
        # _redact_userinfo to find — they're redacted too (uniform with
        # this module's validate_base_url discipline), but that can't be
        # proven through a live trigger the way this one can.
        # -----------------------------------------------------------------
        userinfo_id_path = write_fixture(
            tmpdir,
            "userinfo_id.json",
            {
                "providers": {
                    "https://svc-user:fixture-secret-do-not-print-xyz@gw.example/v1": {
                        "api": "openai-completions",
                        "baseUrl": "https://gw.example/v1",
                    }
                }
            },
        )
        r = run(IMPORT_AND_PRINT_IDS, userinfo_id_path)
        check(
            "credential-bearing id (pasted into the id slot by mistake) is rejected as invalid charset",
            r.returncode != 0 and "invalid" in r.stderr,
            (r.returncode, r.stderr),
        )
        check(
            "...and the embedded password is redacted from the boot-time error, not echoed",
            "fixture-secret-do-not-print-xyz" not in r.stderr and "://***@gw.example" in r.stderr,
            r.stderr,
        )

        valid_charset_path = write_fixture(
            tmpdir,
            "valid_charset.json",
            {
                "providers": {
                    "my-gateway": {"api": "openai-completions", "baseUrl": "https://x.example/v1"},
                    "lm_studio": {"api": "openai-completions", "baseUrl": "https://y.example/v1"},
                }
            },
        )
        r = run(IMPORT_AND_PRINT_IDS, valid_charset_path)
        check(
            "valid ids (letters, digits, '.', '_', '-') still load cleanly, incl. 'lm_studio'",
            r.returncode == 0 and "my-gateway" in r.stdout and "lm_studio" in r.stdout,
            (r.returncode, r.stdout, r.stderr),
        )

        # -----------------------------------------------------------------
        # The Settings card's "Other" tab sentinel (frontend/components/
        # settings/llm-card.tsx's OTHER_TAB) is kept OUT of the id charset
        # ON PURPOSE, rather than the charset excluding underscore to keep
        # it out — see _PROVIDER_ID_RE's module comment. This is the check
        # that ties the two files together: it reads the ACTUAL current
        # value of OTHER_TAB out of the frontend source (not a hardcoded
        # guess that could silently drift from it) and proves the loader
        # rejects a file entry keyed exactly that — so a future edit to
        # either side that reopens the collision (the frontend picking a
        # charset-legal sentinel, or the loader's charset widening to
        # accept '@') fails HERE instead of shipping quietly.
        # -----------------------------------------------------------------
        llm_card_src = (REPO_ROOT / "frontend" / "components" / "settings" / "llm-card.tsx").read_text()
        other_tab_match = re.search(r'const OTHER_TAB = "([^"]+)"', llm_card_src)
        check(
            "found the frontend's OTHER_TAB sentinel declaration to test against",
            other_tab_match is not None,
            llm_card_src[:200],
        )
        if other_tab_match:
            sentinel = other_tab_match.group(1)
            sentinel_path = write_fixture(
                tmpdir,
                "sentinel_collision.json",
                {"providers": {sentinel: {"api": "openai-completions", "baseUrl": "https://x.example/v1"}}},
            )
            r = run(IMPORT_AND_PRINT_IDS, sentinel_path)
            check(
                f"the frontend's OTHER_TAB sentinel ({sentinel!r}) is structurally rejected as a provider id",
                r.returncode != 0 and "invalid" in r.stderr,
                (r.returncode, r.stderr),
            )

        # -----------------------------------------------------------------
        # Two ids that only differ by case/whitespace after normalization
        # — silently colliding (last one wins, discarding the first's
        # shape/base_url/key) is the same "redirects traffic" class the
        # built-in shadow rule exists to prevent. Must be a loud error,
        # not a last-write-wins no-op.
        # -----------------------------------------------------------------
        dup_id_path = write_fixture(
            tmpdir,
            "dup_id.json",
            {
                "providers": {
                    "Groq": {"api": "openai-completions", "baseUrl": "https://api.groq.com/openai/v1", "apiKey": "literal-real-key"},
                    "groq": {"api": "anthropic-messages", "baseUrl": "https://decoy.example/v1"},
                }
            },
        )
        r = run(IMPORT_AND_PRINT_IDS, dup_id_path)
        check(
            "two ids colliding after normalization ('Groq'/'groq') rejected loudly",
            r.returncode != 0 and "duplicate" in r.stderr and "Groq" in r.stderr and "groq" in r.stderr,
            (r.returncode, r.stderr),
        )

        # -----------------------------------------------------------------
        # A directory at the providers path — the exact shape a
        # missing-bind-mount-FILE turns into under Docker — must be a
        # loud error, not a silent no-op (which would reopen the trap the
        # compose design (mounting the directory, not the file) exists to
        # avoid, for anyone overriding DOKKAI_PROVIDERS_FILE directly).
        # -----------------------------------------------------------------
        dir_as_path = os.path.join(tmpdir, "providers_is_a_dir.json")
        os.makedirs(dir_as_path)
        r = run(IMPORT_AND_PRINT_IDS, dir_as_path)
        check(
            "a directory at the providers path is a loud error, not a silent no-op",
            r.returncode != 0 and "directory" in r.stderr,
            (r.returncode, r.stderr),
        )

        # -----------------------------------------------------------------
        # Missing baseUrl — rejected at load time.
        # -----------------------------------------------------------------
        no_base_url_path = write_fixture(
            tmpdir, "no_base_url.json", {"providers": {"groq": {"api": "openai-completions"}}}
        )
        r = run(IMPORT_AND_PRINT_IDS, no_base_url_path)
        check(
            "missing baseUrl: import fails",
            r.returncode != 0 and "baseUrl" in r.stderr,
            (r.returncode, r.stderr),
        )

        # -----------------------------------------------------------------
        # Invalid baseUrl — routed through the SAME validator C2 uses
        # (services.llm_provider.validate_base_url), not a separate,
        # weaker check. Scheme-less and an embedded control character both
        # must be rejected at load time, not fail opaquely later.
        # -----------------------------------------------------------------
        scheme_less_path = write_fixture(
            tmpdir,
            "scheme_less.json",
            {"providers": {"groq": {"api": "openai-completions", "baseUrl": "api.groq.com/openai/v1"}}},
        )
        r = run(IMPORT_AND_PRINT_IDS, scheme_less_path)
        check(
            "scheme-less baseUrl rejected via the shared validator",
            r.returncode != 0 and "http" in r.stderr,
            (r.returncode, r.stderr),
        )

        control_char_path = write_fixture(
            tmpdir,
            "control_char.json",
            {"providers": {"groq": {"api": "openai-completions", "baseUrl": "ht\ntp://api.groq.com/v1"}}},
        )
        r = run(IMPORT_AND_PRINT_IDS, control_char_path)
        check(
            "embedded tab/CR/LF in baseUrl rejected via the shared validator",
            r.returncode != 0 and "tab or line break" in r.stderr,
            (r.returncode, r.stderr),
        )

        # -----------------------------------------------------------------
        # Unknown field in an entry — rejected, which is what actually
        # catches the realistic "apikey" mis-casing typo (a wrong-cased
        # apiKey silently reads back as absent -> keyless otherwise).
        # -----------------------------------------------------------------
        miscased_path = write_fixture(
            tmpdir,
            "miscased.json",
            {
                "providers": {
                    "groq": {
                        "api": "openai-completions",
                        "baseUrl": "https://api.groq.com/openai/v1",
                        "apikey": "${SOME_FIXTURE_VAR}",
                    }
                }
            },
        )
        r = run(IMPORT_AND_PRINT_IDS, miscased_path)
        check(
            "mis-cased 'apikey' field rejected as unknown, not silently keyless",
            r.returncode != 0 and "apikey" in r.stderr and "unknown field" in r.stderr,
            (r.returncode, r.stderr),
        )

        unknown_field_path = write_fixture(
            tmpdir,
            "unknown_field.json",
            {
                "providers": {
                    "groq": {
                        "api": "openai-completions",
                        "baseUrl": "https://api.groq.com/openai/v1",
                        "extraField": "x",
                    }
                }
            },
        )
        r = run(IMPORT_AND_PRINT_IDS, unknown_field_path)
        check(
            "an arbitrary unknown field is rejected too, naming allowed fields",
            r.returncode != 0
            and "extraField" in r.stderr
            and "apiKey" in r.stderr
            and "baseUrl" in r.stderr,
            (r.returncode, r.stderr),
        )

        # -----------------------------------------------------------------
        # Wrong-typed apiKey — loud, like every other malformed field, not
        # a silently keyless provider sending no Authorization header.
        # -----------------------------------------------------------------
        for bad_key_value, label in ((12345, "int"), ({"nested": "x"}, "object"), (["${X}"], "array")):
            bad_type_path = write_fixture(
                tmpdir,
                f"bad_apikey_{label}.json",
                {
                    "providers": {
                        "groq": {
                            "api": "openai-completions",
                            "baseUrl": "https://api.groq.com/openai/v1",
                            "apiKey": bad_key_value,
                        }
                    }
                },
            )
            r = run(IMPORT_AND_PRINT_IDS, bad_type_path)
            check(
                f"wrong-typed apiKey rejected: {label}",
                r.returncode != 0 and "apiKey" in r.stderr and "string" in r.stderr,
                (r.returncode, r.stderr),
            )

        # Explicit "apiKey": null is a JSON author's deliberate choice, not
        # indistinguishable from omission — reject it too (omit the field
        # instead), rather than let it silently mean keyless.
        null_key_path = write_fixture(
            tmpdir,
            "null_apikey.json",
            {"providers": {"groq": {"api": "openai-completions", "baseUrl": "https://api.groq.com/openai/v1", "apiKey": None}}},
        )
        r = run(IMPORT_AND_PRINT_IDS, null_key_path)
        check(
            "explicit apiKey: null rejected (omit the field for keyless)",
            r.returncode != 0 and "apiKey" in r.stderr and "null" in r.stderr,
            (r.returncode, r.stderr),
        )

        # Blank/whitespace-only apiKey — realistic trigger: templating the
        # file with envsubst/sed when the substituted var is unset,
        # producing exactly "". Must be rejected like null, not silently
        # keyless.
        for blank_value in ("", "   "):
            blank_key_path = write_fixture(
                tmpdir,
                f"blank_apikey_{len(blank_value)}.json",
                {"providers": {"groq": {"api": "openai-completions", "baseUrl": "https://api.groq.com/openai/v1", "apiKey": blank_value}}},
            )
            r = run(IMPORT_AND_PRINT_IDS, blank_key_path)
            check(
                f"explicit blank/whitespace apiKey rejected: {blank_value!r}",
                r.returncode != 0 and "apiKey" in r.stderr and "blank" in r.stderr,
                (r.returncode, r.stderr),
            )

        # -----------------------------------------------------------------
        # An apiKey that LOOKS like a botched env reference — reject with
        # a message showing the supported syntax, rather than silently
        # accepting it as a literal secret that 401s later. Discrimination
        # is UPPERCASE-only, matching OpenClaw's own convention exactly
        # (var names match ^[A-Z_][A-Z0-9_]*$): this is what lets a
        # lowercase/punctuated $-leading literal (a
        # real-world token/hash shape) be accepted with NO indirection
        # required, while still catching a genuine typo'd reference
        # attempt. Full matrix: everything the heuristic must catch, and
        # everything it must NOT (a real literal key must never be
        # rejected).
        # -----------------------------------------------------------------
        must_reject = [
            "$GROQ_API_KEY",             # bare $VAR (whole string, uppercase), no braces
            "Bearer ${GROQ_API_KEY}",    # wrapped in other text
            "${}",                       # empty var name
            "${ VAR }",                  # spaces inside braces
            "${VAR}${VAR2}",             # multiple substitutions
            "${VAR:-default}",           # shell-style default, unsupported
            "$${VAR}",                   # doubled $
            "${9VAR}",                   # var name can't start with a digit
            "${my_key}",                 # lowercase name — env vars are UPPERCASE by convention
        ]
        for i, bad_ref in enumerate(must_reject):
            lookalike_path = write_fixture(
                tmpdir,
                f"lookalike_reject_{i}.json",
                {"providers": {"groq": {"api": "openai-completions", "baseUrl": "https://api.groq.com/openai/v1", "apiKey": bad_ref}}},
            )
            r = run(IMPORT_AND_PRINT_IDS, lookalike_path)
            check(
                f"env-reference-lookalike apiKey rejected: {bad_ref!r}",
                r.returncode != 0 and "${VAR_NAME}" in r.stderr,
                (r.returncode, r.stderr),
            )
            # BLOCKING regression check: the raw value must NEVER be echoed
            # into the boot traceback (uvicorn prints exceptions to
            # docker compose logs / CI output / wherever a user pastes it
            # next) — only its (non-secret) length, and the ${VAR_NAME}
            # escape hatch must be pointed at explicitly.
            check(
                f"env-reference-lookalike rejection never echoes the raw value: {bad_ref!r}",
                bad_ref not in r.stderr,
                r.stderr,
            )
            check(
                f"env-reference-lookalike rejection names the environment-variable escape hatch: {bad_ref!r}",
                "environment variable" in r.stderr,
                r.stderr,
            )

        must_accept = [
            "sk-abc$def",
            "$",
            "$$",
            "$1abc",
            "%VAR%",
            # Real-world $-leading literal shapes — a $ followed by
            # lowercase/punctuation is unambiguously NOT an env reference
            # once the discriminator is "env vars are UPPERCASE", so
            # these now need no ${VAR} indirection at all (the earlier,
            # case-insensitive heuristic used to false-reject both).
            "$ecretLiteralKeyAbc123",
            "$argon2id$v=19$m=1",
            # Uppercase-leading but not a clean whole-string identifier
            # (hyphens aren't valid in an env var name) — the
            # bare-shorthand check is anchored at BOTH ends, matching
            # OpenClaw's own convention of anchoring the pattern rather
            # than just checking "starts with $ + uppercase".
            "$SECRET-123-SUFFIX",
        ]
        for i, ok_ref in enumerate(must_accept):
            accept_path = write_fixture(
                tmpdir,
                f"lookalike_accept_{i}.json",
                {"providers": {"groq": {"api": "openai-completions", "baseUrl": "https://api.groq.com/openai/v1", "apiKey": ok_ref}}},
            )
            r = run(
                "import services.llm_provider as lp\nprint(lp.get_provider('groq')._client.api_key)",
                accept_path,
            )
            check(
                f"NOT rejected as a lookalike, used as a literal: {ok_ref!r}",
                r.returncode == 0 and r.stdout.strip() == ok_ref,
                (r.returncode, r.stdout, r.stderr),
            )

        # A genuinely literal key that happens to be a normal-looking
        # string (no '$') must still work — the heuristic above must not
        # over-trigger on ordinary keys.
        genuinely_literal_path = write_fixture(
            tmpdir,
            "genuinely_literal.json",
            {"providers": {"groq": {"api": "openai-completions", "baseUrl": "https://api.groq.com/openai/v1", "apiKey": "gsk_fixture_literal_abc123"}}},
        )
        r = run(
            "import services.llm_provider as lp\nprint(lp.get_provider('groq')._client.api_key)",
            genuinely_literal_path,
        )
        check(
            "a genuinely literal key (no '$') still works",
            r.returncode == 0 and r.stdout.strip() == "gsk_fixture_literal_abc123",
            (r.returncode, r.stdout, r.stderr),
        )

        # A literal that IS still rejected directly (the bare-uppercase-
        # shorthand shape, "$ALLCAPS...", exactly the "$GROQ_API_KEY"
        # case) still has a working escape hatch: put it in an env var,
        # reference that var with ${VAR_NAME}. Proves the message's advice
        # is actually followable, not a dead end — env var VALUES are
        # used verbatim, with no re-interpretation of the leading '$'.
        dollar_leading_env_path = write_fixture(
            tmpdir,
            "dollar_leading_via_env.json",
            {"providers": {"groq": {"api": "openai-completions", "baseUrl": "https://api.groq.com/openai/v1", "apiKey": "${DOLLAR_LEADING_FIXTURE_VAR}"}}},
        )
        r = run(
            "import services.llm_provider as lp\nprint(lp.get_provider('groq')._client.api_key)",
            dollar_leading_env_path,
            extra_env={"DOLLAR_LEADING_FIXTURE_VAR": "$ALLCAPS_LOOKS_LIKE_SHORTHAND"},
        )
        check(
            "a literal that WOULD be rejected directly ('$ALLCAPS...', bare-shorthand shape) works when routed through ${VAR_NAME}",
            r.returncode == 0 and r.stdout.strip() == "$ALLCAPS_LOOKS_LIKE_SHORTHAND",
            (r.returncode, r.stdout, r.stderr),
        )
        # Control: that same value, used DIRECTLY (no indirection), must
        # still be rejected — confirms the escape-hatch test above is
        # actually proving something (the value really would 401 opaquely
        # if silently accepted as a literal, since it also happens to be
        # syntactically indistinguishable from the shorthand this loader
        # deliberately doesn't support).
        direct_reject_path = write_fixture(
            tmpdir,
            "dollar_leading_direct.json",
            {"providers": {"groq": {"api": "openai-completions", "baseUrl": "https://api.groq.com/openai/v1", "apiKey": "$ALLCAPS_LOOKS_LIKE_SHORTHAND"}}},
        )
        r = run(IMPORT_AND_PRINT_IDS, direct_reject_path)
        check(
            "that same value used directly (no ${VAR} indirection) is still rejected",
            r.returncode != 0 and "${VAR_NAME}" in r.stderr,
            (r.returncode, r.stderr),
        )

        # -----------------------------------------------------------------
        # A valid provider is loaded and shows up as "registered" (part of
        # get_registered_provider_ids()), not as an anonymous custom id.
        # -----------------------------------------------------------------
        groq_path = write_fixture(
            tmpdir,
            "groq.json",
            {
                "providers": {
                    "groq": {
                        "api": "openai-completions",
                        "baseUrl": "https://api.groq.com/openai/v1",
                        "apiKey": "${TEST_GROQ_FIXTURE_KEY}",
                        "models": ["llama-3.3-70b-versatile"],
                    }
                }
            },
        )
        r = run(
            IMPORT_AND_PRINT_IDS,
            groq_path,
            extra_env={"TEST_GROQ_FIXTURE_KEY": "fixture-secret-do-not-print-abc123"},
        )
        check(
            "file-registered id joins get_registered_provider_ids()",
            r.returncode == 0 and "groq" in r.stdout,
            (r.returncode, r.stdout, r.stderr),
        )

        # -----------------------------------------------------------------
        # get_builtin_provider_ids()/display_name_for()/default_base_url_for()
        # (feature 22, C5 review) — the accessors controllers.config's
        # provider-listing endpoint uses instead of a second, driftable
        # copy of the three built-in ids / a str.title()-mangled label /
        # a hardcoded Gemini URL.
        # -----------------------------------------------------------------
        accessors_probe = (
            "import services.llm_provider as lp\n"
            "print(sorted(lp.get_builtin_provider_ids()))\n"
            "print('ollama' in lp.get_builtin_provider_ids())\n"
            "print('groq' in lp.get_builtin_provider_ids())\n"
            "print(lp.display_name_for('openai'))\n"
            "print(lp.display_name_for('groq'))\n"
            "print(lp.display_name_for('not-registered-at-all'))\n"
            "print(lp.default_base_url_for('gemini'))\n"
            "print(lp.default_base_url_for('openai'))\n"
            "print(lp.default_base_url_for('groq'))\n"
        )
        r = run(accessors_probe, groq_path, extra_env={"TEST_GROQ_FIXTURE_KEY": "fixture-secret-do-not-print-abc123"})
        out_lines = r.stdout.strip().splitlines()
        check(
            "accessors probe runs cleanly (9 output lines)",
            r.returncode == 0 and len(out_lines) == 9,
            (r.returncode, r.stdout, r.stderr),
        )
        if len(out_lines) == 9:
            check(
                "get_builtin_provider_ids() is exactly the three built-ins",
                out_lines[0] == "['anthropic', 'gemini', 'openai']",
                out_lines[0],
            )
            check("get_builtin_provider_ids() excludes ollama", out_lines[1] == "False", out_lines[1])
            check(
                "get_builtin_provider_ids() excludes a file-registered id",
                out_lines[2] == "False",
                out_lines[2],
            )
            check("display_name_for('openai') is 'OpenAI'", out_lines[3] == "OpenAI", out_lines[3])
            check(
                "display_name_for('groq') is the file's own id, not a title-cased mangling",
                out_lines[4] == "groq",
                out_lines[4],
            )
            check(
                "display_name_for(unregistered) is None",
                out_lines[5] == "None",
                out_lines[5],
            )
            check(
                "default_base_url_for('gemini') is its real, non-guessable default",
                out_lines[6] == "https://generativelanguage.googleapis.com/v1beta/openai/",
                out_lines[6],
            )
            check(
                "default_base_url_for('openai') is None (no override — SDK's own default applies)",
                out_lines[7] == "None",
                out_lines[7],
            )
            check(
                "default_base_url_for('groq') is None EVEN THOUGH groq has a real baseUrl of its own "
                "(file-registered defaults are never exposed through this accessor)",
                out_lines[8] == "None",
                out_lines[8],
            )

        # -----------------------------------------------------------------
        # Three independent copies of the built-in provider id list exist
        # (feature 22): this module's `_REGISTRY` (the source of truth,
        # reached here through `get_builtin_provider_ids()`), the CLI's
        # `BUILTIN_DESCRIPTOR_PROVIDERS` (`cli/src/lib/descriptor.ts`),
        # and the frontend's degraded-mode `FALLBACK_BUILTIN_PROVIDERS`
        # (`frontend/components/settings/llm-card.tsx`). Each carries an
        # honest "this can go stale" comment and degrades safely — that
        # part is fine and stays — but the `OTHER_TAB` sentinel got a
        # dedicated cross-language test (above) tying the frontend and
        # the loader together, while the built-in list itself never did:
        # adding a fourth built-in to `_REGISTRY` alone would desync both
        # copies with zero failing tests. This reads the id set out of
        # ALL THREE sources' ACTUAL current text — never a hardcoded
        # guess that could itself drift from any of them, the same
        # discipline the `OTHER_TAB` check above already uses — and
        # proves they agree, so a future built-in added to one without
        # the matching edit to the other two fails HERE instead of
        # shipping a silently-stale CLI table or Settings fallback.
        # -----------------------------------------------------------------
        registry_probe = "import services.llm_provider as lp\nprint(sorted(lp.get_builtin_provider_ids()))\n"
        r = run(registry_probe, None)
        check(
            "reading the real _REGISTRY's built-in id set to compare the two copies against",
            r.returncode == 0,
            (r.returncode, r.stdout, r.stderr),
        )
        registry_builtin_ids: set[str] = set()
        if r.returncode == 0:
            registry_builtin_ids = set(ast.literal_eval(r.stdout.strip()))

        llm_card_src = (REPO_ROOT / "frontend" / "components" / "settings" / "llm-card.tsx").read_text()
        frontend_block_match = re.search(
            r"const FALLBACK_BUILTIN_PROVIDERS[^=]*=\s*\[(.*?)\];", llm_card_src, re.DOTALL
        )
        check(
            "found the frontend's FALLBACK_BUILTIN_PROVIDERS declaration to test against",
            frontend_block_match is not None,
            llm_card_src[:200],
        )
        if frontend_block_match:
            frontend_ids = set(re.findall(r'id:\s*"([a-z0-9._-]+)"', frontend_block_match.group(1)))
            check(
                "frontend FALLBACK_BUILTIN_PROVIDERS ids match the real _REGISTRY built-ins exactly",
                frontend_ids == registry_builtin_ids,
                (sorted(frontend_ids), sorted(registry_builtin_ids)),
            )

        descriptor_ts_src = (REPO_ROOT / "cli" / "src" / "lib" / "descriptor.ts").read_text()
        cli_block_match = re.search(
            r"const BUILTIN_DESCRIPTOR_PROVIDERS[^=]*=\s*new Map[^(]*\(\[(.*?)\]\);", descriptor_ts_src, re.DOTALL
        )
        check(
            "found the CLI's BUILTIN_DESCRIPTOR_PROVIDERS declaration to test against",
            cli_block_match is not None,
            descriptor_ts_src[:200],
        )
        if cli_block_match:
            cli_ids = set(re.findall(r'\[\s*"([a-z0-9._-]+)"\s*,\s*\{\s*keyEnv:', cli_block_match.group(1)))
            check(
                "CLI BUILTIN_DESCRIPTOR_PROVIDERS ids match the real _REGISTRY built-ins exactly",
                cli_ids == registry_builtin_ids,
                (sorted(cli_ids), sorted(registry_builtin_ids)),
            )

        # -----------------------------------------------------------------
        # default_base_url_for()'s _redact_userinfo pass is a defense-in-
        # depth belt-and-braces measure (feature 22, C5 review) — no
        # built-in default carries embedded userinfo today, so this can't
        # be proven through the real registry. White-box: replace a
        # built-in's spec (frozen dataclass -> dataclasses.replace) with
        # one whose default_base_url legitimately embeds Basic-auth
        # credentials, and confirm the accessor still scrubs them, rather
        # than assuming the wrapping call is dead code.
        # -----------------------------------------------------------------
        redact_probe = (
            "import dataclasses\n"
            "import services.llm_provider as lp\n"
            "spec = lp._REGISTRY['openai']\n"
            "lp._REGISTRY['openai'] = dataclasses.replace(\n"
            "    spec, default_base_url='https://svc-user:svc-secret@sneaky.example/v1'\n"
            ")\n"
            "print(lp.default_base_url_for('openai'))\n"
        )
        r = run(redact_probe, None)
        check(
            "default_base_url_for() redacts embedded userinfo, not just built-in-only gating",
            r.returncode == 0
            and "svc-secret" not in r.stdout
            and "://***@sneaky.example/v1" in r.stdout,
            (r.returncode, r.stdout, r.stderr),
        )

        # -----------------------------------------------------------------
        # ${ENV} resolved: get_provider(id) with NO explicit api_key/base_url
        # succeeds and uses the registry's own resolved key/base_url.
        # -----------------------------------------------------------------
        resolved_probe = (
            "import services.llm_provider as lp\n"
            "p = lp.get_provider('groq')\n"
            "print(type(p).__name__)\n"
            "print(p._client.api_key)\n"
            "print(str(p._client.base_url))\n"
        )
        r = run(resolved_probe, groq_path, extra_env={"TEST_GROQ_FIXTURE_KEY": "fixture-secret-do-not-print-abc123"})
        out_lines = r.stdout.strip().splitlines()
        check(
            "get_provider('groq') with no explicit key/base_url succeeds",
            r.returncode == 0 and len(out_lines) == 3,
            (r.returncode, r.stdout, r.stderr),
        )
        if len(out_lines) == 3:
            check(
                "get_provider('groq') resolves the ${ENV}-interpolated key",
                out_lines[1] == "fixture-secret-do-not-print-abc123",
                out_lines,
            )
            check(
                "get_provider('groq') resolves the file's default base_url",
                out_lines[2].rstrip("/") == "https://api.groq.com/openai/v1",
                out_lines,
            )

        # -----------------------------------------------------------------
        # THE CORE FIX: ${ENV} resolution is LAZY (at get_provider()/
        # key_env_for() USE time), not eager at import/load time.
        #
        # This directly reproduces and closes the reported bug: main.py
        # imports the controllers (-> llm_config -> llm_provider, which
        # triggers the providers.json load) BEFORE calling load_dotenv() —
        # so a var that only lives in .env is provably absent from
        # os.environ at the moment this module is first imported. The
        # fixture var below is (a) absent from the subprocess's
        # environment at LAUNCH, and only (b) set from WITHIN the
        # process, AFTER `import services.llm_provider` has already run —
        # exactly mirroring main.py's import order. If resolution were
        # still eager (at import time), get_provider() would keep raising
        # even after the var is set later in the same process, since
        # nothing would re-check the environment.
        #
        # This also proves the related acceptance point: a provider whose
        # key is exported AFTER boot works on the very next call, with no
        # restart — because resolution re-checks os.environ every time,
        # never caching a value (or its absence) from load time.
        # -----------------------------------------------------------------
        lazy_env_var = "LAZY_FIXTURE_VAR_NEVER_PRESET"
        lazy_path = write_fixture(
            tmpdir,
            "lazy.json",
            {"providers": {"groq": {"api": "openai-completions", "baseUrl": "https://api.groq.com/openai/v1", "apiKey": f"${{{lazy_env_var}}}"}}},
        )
        lazy_probe = (
            "import os\n"
            f"assert os.environ.get({lazy_env_var!r}) is None, "
            "'test setup bug: fixture var must be absent at process launch'\n"
            "import services.llm_provider as lp  # import happens BEFORE the var is ever set\n"
            "try:\n"
            "    lp.get_provider('groq')\n"
            "    before_result = 'NO_ERROR'\n"
            "except ValueError:\n"
            "    before_result = 'ERROR'\n"
            f"before_key_env = lp.key_env_for('groq')\n"
            "# Simulate load_dotenv() running late / the var being exported after boot:\n"
            f"os.environ[{lazy_env_var!r}] = 'lazy-secret-do-not-print-xyz789'\n"
            "p = lp.get_provider('groq')  # same process, NO restart\n"
            "after_key_matches = (p._client.api_key == 'lazy-secret-do-not-print-xyz789')\n"
            "after_key_env = lp.key_env_for('groq')\n"
            "print('before_result:', before_result)\n"
            "print('before_key_env:', before_key_env)\n"
            "print('after_key_matches:', after_key_matches)\n"
            "print('after_key_env:', after_key_env)\n"
        )
        # Crucially: extra_env does NOT set lazy_env_var — it must be
        # genuinely absent from the child's environment at launch.
        r = run(lazy_probe, lazy_path)
        check(
            "lazy resolution: import succeeds before the var is ever set",
            r.returncode == 0,
            (r.returncode, r.stdout, r.stderr),
        )
        check(
            "lazy resolution: get_provider() raises BEFORE the var is set (import-time snapshot would wrongly succeed OR permanently fail)",
            "before_result: ERROR" in r.stdout,
            r.stdout,
        )
        check(
            "lazy resolution: key_env_for() is mandatory BEFORE the var is set",
            f"before_key_env: {lazy_env_var}" in r.stdout,
            r.stdout,
        )
        check(
            "lazy resolution: get_provider() succeeds AFTER the var is set, same process, no restart",
            "after_key_matches: True" in r.stdout,
            r.stdout,
        )
        check(
            "lazy resolution: key_env_for() is no longer mandatory AFTER the var is set",
            "after_key_env: None" in r.stdout,
            r.stdout,
        )
        check(
            "lazy resolution: the resolved secret value is never printed",
            "lazy-secret-do-not-print-xyz789" not in r.stdout and "lazy-secret-do-not-print-xyz789" not in r.stderr,
            (r.stdout, r.stderr),
        )

        # -----------------------------------------------------------------
        # Explicit api_key/base_url still override the file's registered
        # defaults (existing get_provider precedent, preserved).
        # -----------------------------------------------------------------
        override_probe = (
            "import services.llm_provider as lp\n"
            "p = lp.get_provider('groq', api_key='override-key', base_url='http://localhost:9999/v1')\n"
            "print(p._client.api_key)\n"
            "print(str(p._client.base_url))\n"
        )
        r = run(override_probe, groq_path, extra_env={"TEST_GROQ_FIXTURE_KEY": "fixture-secret-do-not-print-abc123"})
        out_lines = r.stdout.strip().splitlines()
        check(
            "explicit api_key/base_url override the file's registered defaults",
            r.returncode == 0
            and len(out_lines) == 2
            and out_lines[0] == "override-key"
            and out_lines[1].rstrip("/") == "http://localhost:9999/v1",
            (r.returncode, r.stdout, r.stderr),
        )

        # -----------------------------------------------------------------
        # ${ENV} unresolved (var not set, ever): import still succeeds (a
        # provider whose key hasn't been exported yet must not block boot
        # for every other provider), but get_provider(id) raises a precise
        # diagnostic naming the variable — and key_env_for(id) reports the
        # key as mandatory (forcing an explicit override) until it's fixed.
        # -----------------------------------------------------------------
        unresolved_path = write_fixture(
            tmpdir,
            "unresolved.json",
            {
                "providers": {
                    "groq": {
                        "api": "openai-completions",
                        "baseUrl": "https://api.groq.com/openai/v1",
                        "apiKey": "${TEST_GROQ_NEVER_SET_FIXTURE_VAR}",
                    }
                }
            },
        )
        r = run(IMPORT_AND_PRINT_IDS, unresolved_path)
        check(
            "unresolved ${ENV} key: import still succeeds (doesn't block boot)",
            r.returncode == 0 and "groq" in r.stdout,
            (r.returncode, r.stdout, r.stderr),
        )

        unresolved_use = (
            "import services.llm_provider as lp\n"
            "try:\n"
            "    lp.get_provider('groq')\n"
            "    print('NO_ERROR')\n"
            "except ValueError as e:\n"
            "    print(f'ERROR: {e}')\n"
        )
        r = run(unresolved_use, unresolved_path)
        check(
            "unresolved ${ENV} key: get_provider() raises at use time",
            r.returncode == 0 and "ERROR:" in r.stdout,
            (r.returncode, r.stdout, r.stderr),
        )
        check(
            "unresolved ${ENV} key: diagnostic names the variable",
            "TEST_GROQ_NEVER_SET_FIXTURE_VAR" in r.stdout,
            r.stdout,
        )
        # The diagnostic must name the REAL file that produced this spec
        # (here, DOKKAI_PROVIDERS_FILE's actual scratch path — nowhere
        # near "config/providers.json") — not a hardcoded default that
        # sends the user to edit a file that isn't the problem.
        check(
            "unresolved ${ENV} key: diagnostic names the ACTUAL file path (DOKKAI_PROVIDERS_FILE override), not a hardcoded default",
            unresolved_path in r.stdout and "config/providers.json" not in r.stdout,
            (unresolved_path, r.stdout),
        )
        # Nothing to leak (the var was never set), but confirm the
        # diagnostic doesn't echo an empty-string key value either.
        check(
            "unresolved ${ENV} key: no key/value material printed",
            "fixture-secret" not in r.stdout and "fixture-secret" not in r.stderr,
            (r.stdout, r.stderr),
        )

        key_env_probe = (
            "import services.llm_provider as lp\n"
            "print(lp.key_env_for('groq'))\n"
        )
        r = run(key_env_probe, unresolved_path)
        check(
            "unresolved ${ENV} key: key_env_for() reports it mandatory (names the var)",
            r.returncode == 0 and r.stdout.strip() == "TEST_GROQ_NEVER_SET_FIXTURE_VAR",
            (r.returncode, r.stdout, r.stderr),
        )

        # -----------------------------------------------------------------
        # Literal key (no ${...} syntax) — accepted as-is.
        # -----------------------------------------------------------------
        literal_path = write_fixture(
            tmpdir,
            "literal.json",
            {
                "providers": {
                    "groq": {
                        "api": "openai-completions",
                        "baseUrl": "https://api.groq.com/openai/v1",
                        "apiKey": "literal-fixture-key-xyz",
                    }
                }
            },
        )
        r = run(
            "import services.llm_provider as lp\nprint(lp.get_provider('groq')._client.api_key)",
            literal_path,
        )
        check(
            "literal apiKey (no ${...}) is used as-is",
            r.returncode == 0 and r.stdout.strip() == "literal-fixture-key-xyz",
            (r.returncode, r.stdout, r.stderr),
        )

        # -----------------------------------------------------------------
        # Keyless entry (no apiKey at all) — a local server needing no key.
        # -----------------------------------------------------------------
        keyless_path = write_fixture(
            tmpdir,
            "keyless.json",
            {"providers": {"localvllm": {"api": "openai-completions", "baseUrl": "http://localhost:8000/v1"}}},
        )
        r = run(
            "import services.llm_provider as lp\n"
            "p = lp.get_provider('localvllm')\n"
            "print('OK', repr(p._client.api_key))\n"
            "print('key_env_for:', lp.key_env_for('localvllm'))\n",
            keyless_path,
        )
        check(
            "keyless file-registered provider: get_provider() succeeds without a key",
            r.returncode == 0 and r.stdout.startswith("OK"),
            (r.returncode, r.stdout, r.stderr),
        )
        check(
            "keyless file-registered provider: key not mandatory (key_env_for is None)",
            "key_env_for: None" in r.stdout,
            r.stdout,
        )

        # -----------------------------------------------------------------
        # Built-ins are untouched: still key-mandatory, still resolve
        # exactly as before, regardless of an unrelated file entry.
        # -----------------------------------------------------------------
        builtin_probe = (
            "import services.llm_provider as lp\n"
            "try:\n"
            "    lp.get_provider('openai')\n"
            "    print('NO_ERROR')\n"
            "except ValueError as e:\n"
            "    print(f'ERROR: {e}')\n"
        )
        r = run(builtin_probe, groq_path, extra_env={"TEST_GROQ_FIXTURE_KEY": "fixture-secret-do-not-print-abc123"})
        check(
            "built-in 'openai' still requires an explicit key with a file present",
            r.returncode == 0 and "requires an API key" in r.stdout,
            (r.returncode, r.stdout, r.stderr),
        )

        # -----------------------------------------------------------------
        # models[] parsed and validated (type-checked).
        # -----------------------------------------------------------------
        bad_models_path = write_fixture(
            tmpdir,
            "bad_models.json",
            {
                "providers": {
                    "groq": {
                        "api": "openai-completions",
                        "baseUrl": "https://api.groq.com/openai/v1",
                        "models": "not-a-list",
                    }
                }
            },
        )
        r = run(IMPORT_AND_PRINT_IDS, bad_models_path)
        check(
            "non-list 'models' field rejected at load time",
            r.returncode != 0 and "models" in r.stderr,
            (r.returncode, r.stderr),
        )

        # -----------------------------------------------------------------
        # models[] is WIRED, not discarded: it becomes this id's
        # list_models() fallback (used when the live catalog call fails) —
        # the same mechanism _FALLBACK_MODELS already provides for the
        # three built-ins.
        # -----------------------------------------------------------------
        r = run(
            "import services.llm_provider as lp\nprint(lp._FALLBACK_MODELS.get('groq'))",
            groq_path,
            extra_env={"TEST_GROQ_FIXTURE_KEY": "fixture-secret-do-not-print-abc123"},
        )
        check(
            "models[] is wired into _FALLBACK_MODELS (the list_models() fallback)",
            r.returncode == 0 and r.stdout.strip() == "['llama-3.3-70b-versatile']",
            (r.returncode, r.stdout, r.stderr),
        )

        # An empty/absent models[] must NOT poison _FALLBACK_MODELS with an
        # empty list for that id (that would make list_models() return []
        # instead of leaving the id simply absent from the fallback dict).
        r = run(
            "import services.llm_provider as lp\nprint('localvllm' in lp._FALLBACK_MODELS)",
            keyless_path,
        )
        check(
            "empty/absent models[] does not add an empty fallback entry",
            r.returncode == 0 and r.stdout.strip() == "False",
            (r.returncode, r.stdout, r.stderr),
        )

        # -----------------------------------------------------------------
        # repr=False on _ProviderSpec.default_api_key: an incidental
        # repr()/log of the spec object must never print a literal key.
        # -----------------------------------------------------------------
        r = run(
            "import services.llm_provider as lp\n"
            "spec = lp._REGISTRY['groq']\n"
            "print(repr(spec))\n",
            literal_path,
        )
        check(
            "repr(_ProviderSpec) never includes the literal key value",
            r.returncode == 0 and "literal-fixture-key-xyz" not in r.stdout,
            (r.returncode, r.stdout, r.stderr),
        )
        check(
            "repr(_ProviderSpec) still shows non-secret fields (sanity: repr isn't just empty/broken)",
            "shape=" in r.stdout and "display_name=" in r.stdout,
            r.stdout,
        )

        # -----------------------------------------------------------------
        # End-to-end through the CONFIG LAYER (services.llm_config): a
        # file-registered id needs neither base_url nor key from the POST
        # body (get_registered_provider_ids()/key_env_for() treat it as
        # registered, not anonymous-custom), and the persisted config
        # round-trips into get_active_provider() correctly. health_check
        # is monkeypatched — this test makes zero live API calls.
        # -----------------------------------------------------------------
        config_layer_probe = (
            "import asyncio\n"
            "import services.llm_config as llm_config\n"
            "\n"
            "class _StubProvider:\n"
            "    async def health_check(self, model=None):\n"
            "        return True, 'ok'\n"
            "\n"
            "_calls = []\n"
            "def _stub_get_provider(name, *, api_key='', base_url=None):\n"
            "    _calls.append((name, api_key, base_url))\n"
            "    return _StubProvider()\n"
            "\n"
            "async def _noop_save(config):\n"
            "    return None\n"
            "\n"
            "llm_config.get_provider = _stub_get_provider\n"
            "llm_config.save_persisted_config = _noop_save\n"
            "\n"
            "async def main():\n"
            "    ok, msg = await llm_config.validate_and_save_config(\n"
            "        is_local=False, provider_name='groq', model='llama-3.3-70b-versatile', key='',\n"
            "    )\n"
            "    print('ok:', ok, 'msg:', msg)\n"
            "    persisted = llm_config.get_config_store().get_llm_config()\n"
            "    print('persisted_provider:', persisted.provider_name)\n"
            "    print('persisted_base_url:', persisted.base_url)\n"
            "    print('persisted_key_empty:', persisted.key == '')\n"
            "    provider, model = llm_config.get_active_provider()\n"
            "    print('active_model:', model)\n"
            "    print('calls:', _calls)\n"
            "\n"
            "asyncio.run(main())\n"
        )
        r = run(config_layer_probe, groq_path, extra_env={"TEST_GROQ_FIXTURE_KEY": "fixture-secret-do-not-print-abc123"})
        check(
            "config layer: file-registered id accepted with no base_url/key in the POST",
            r.returncode == 0 and "ok: True" in r.stdout,
            (r.returncode, r.stdout, r.stderr),
        )
        check(
            "config layer: get_active_provider() resolves after persisting an empty base_url/key",
            "active_model: llama-3.3-70b-versatile" in r.stdout,
            r.stdout,
        )
        check(
            "config layer round trip: never leaks the fixture secret to stdout/stderr",
            "fixture-secret-do-not-print-abc123" not in r.stdout
            and "fixture-secret-do-not-print-abc123" not in r.stderr,
            (r.stdout, r.stderr),
        )

        # -----------------------------------------------------------------
        # Config layer still rejects a genuinely unknown (non-registered,
        # non-custom) id without a base_url — the file doesn't turn
        # EVERYTHING into a registered id, only the ids it declares.
        # -----------------------------------------------------------------
        unknown_probe = (
            "import asyncio\n"
            "import services.llm_config as llm_config\n"
            "\n"
            "class _StubProvider:\n"
            "    async def health_check(self, model=None):\n"
            "        return True, 'ok'\n"
            "\n"
            "def _stub_get_provider(name, *, api_key='', base_url=None):\n"
            "    return _StubProvider()\n"
            "\n"
            "async def _noop_save(config):\n"
            "    return None\n"
            "\n"
            "llm_config.get_provider = _stub_get_provider\n"
            "llm_config.save_persisted_config = _noop_save\n"
            "\n"
            "async def main():\n"
            "    ok, msg = await llm_config.validate_and_save_config(\n"
            "        is_local=False, provider_name='totally-unregistered', model='m', key='',\n"
            "    )\n"
            "    print('ok:', ok, 'msg:', msg)\n"
            "\n"
            "asyncio.run(main())\n"
        )
        r = run(unknown_probe, groq_path, extra_env={"TEST_GROQ_FIXTURE_KEY": "fixture-secret-do-not-print-abc123"})
        check(
            "config layer: an id NOT in the file still needs a base_url (not silently registered)",
            r.returncode == 0 and "ok: False" in r.stdout and "base_url" in r.stdout,
            (r.returncode, r.stdout, r.stderr),
        )

    print(f"\n{_passed} passed, {_failed} failed")
    if _failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
