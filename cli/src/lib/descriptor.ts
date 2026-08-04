/**
 * Shared descriptor-provider resolution (feature 22, C6).
 *
 * `doctor`, `status`, and `up` all read `DESC_PROVIDER`/`DESC_MODEL`/
 * `DESC_BASE_URL`/`DESC_API_KEY` and need to answer the same question: is
 * the descriptor local (Ollama — keeps its own byte-identical `ollama
 * pull` path in each command) or remote? This module is the single place
 * that answers it — each command owns only how it RENDERS the answer (a
 * `[ok]`/`[info]`/`[warn]` table for `doctor`, colored inline lines for
 * `status`/`up`), never how it's derived. Extracted after three commands
 * started duplicating this logic.
 *
 * Governing principle (adversarial review, feature 22 C6, round 2): **the
 * CLI reports what it can observe; the API is the authority on
 * validity.** This module never makes an affirmative claim it can't fully
 * back — no restated URL-format validation (`services.llm_provider.
 * validate_base_url` stays server-side, same reasoning as deleting the
 * browser-side URL validator in C5: a check that restates server logic in
 * another language drifts, and a confidently wrong diagnostic is worse
 * than none, especially in the command people run when things are
 * already broken), no restated `config/providers.json` per-entry
 * validation (id charset, built-in shadowing, duplicate ids, an entry's
 * `api`/`baseUrl`/`apiKey` shape — all of that already fails loudly at
 * API boot, `services.llm_provider._load_providers_file`; this module
 * only checks whether the file PARSES and has the right TOP-LEVEL shape,
 * and whether an id exists as a JSON key — never whether an entry's
 * fields are individually well-formed). A finding is "ok" only for a
 * fact fully within this module's own power to determine (an env var is
 * set, a JSON key exists, JSON parses); it flags a *definite* failure
 * (unparseable JSON, wrong top-level shape) as a warning; it never
 * asserts something will work.
 *
 * Never makes a network call. Never prints a key VALUE — only presence.
 * Any admin-authored string that could embed a credential (a provider id
 * pasted from a URL, a base URL with userinfo) is redacted before it's
 * ever put in a message — see `redactUserinfo`, applied uniformly,
 * mirroring `services.llm_provider._redact_userinfo`'s own discipline.
 */
import { existsSync, readFileSync, statSync } from "node:fs";
import { isAbsolute, join } from "node:path";
import { envVar } from "./config.js";

// "ok" is reserved for a claim this module can fully back (an env var is
// set, a value is non-empty) that's also genuinely a positive signal.
// "info" is a neutral observation this module made no judgment call
// on — an id exists as a JSON key, a URL is present — where "ok" would
// overstate what was actually checked (URL FORMAT, an entry's other
// fields, ... are the API's job; see the module docstring). "warning" is
// a definite, fully-determinable problem.
export type FindingLevel = "ok" | "warning" | "info";

export interface Finding {
  level: FindingLevel;
  message: string;
}

function finding(level: FindingLevel, message: string): Finding {
  return { level, message };
}

/** `DESC_PROVIDER`, defaulting to `ollama` — matches `describe.py`'s own default exactly. */
export function resolveDescriptorProvider(): string {
  return envVar("DESC_PROVIDER", "ollama").toLowerCase().trim();
}

/**
 * Whether `descProvider` is the local Ollama descriptor — the ONLY case
 * where `DESC_MODEL` should be checked against Ollama's installed models
 * / suggested via `ollama pull`. Every other id is remote (built-in, file-
 * registered, or fully custom).
 */
export function isOllamaDescriptor(descProvider: string): boolean {
  return descProvider === "ollama";
}

// Restated from `src/services/llm_provider.py`'s `_REGISTRY` — an
// unavoidable cross-language duplication (this module needs to know
// which env var to look for), but see `checkDescriptorRemote`'s "not
// recognized" branch: an id NOT in this table is reported as "this CLI
// doesn't recognize it", never as "the descriptor has no way to reach
// it" — this table going stale (Python gaining a fourth built-in) must
// not turn into a false failure claim.
//
// A `Map`, not a plain object: `DESC_PROVIDER=constructor` (or
// `__proto__`, `toString`, ...) indexed into a plain object literal
// resolves through the PROTOTYPE CHAIN to a truthy built-in value,
// producing a garbage diagnostic (`[warn] undefined requires an API
// key — set DESC_API_KEY or undefined`) instead of Python's own
// `Unknown provider 'constructor'` — precisely the confidently-wrong
// diagnostic this module exists to avoid, for admittedly absurd input.
// `Map.get` has no prototype chain to fall through.
const BUILTIN_DESCRIPTOR_PROVIDERS = new Map<string, { keyEnv: string; displayName: string }>([
  ["openai", { keyEnv: "OPENAI_API_KEY", displayName: "OpenAI" }],
  ["gemini", { keyEnv: "GEMINI_API_KEY", displayName: "Gemini" }],
  ["anthropic", { keyEnv: "ANTHROPIC_API_KEY", displayName: "Anthropic" }],
]);

/**
 * Redact every embedded `user:pass@` credential from a URL (or URL-shaped
 * string) before printing it — the `g` flag matters: without it only the
 * FIRST match is redacted, so a second occurrence (e.g. credentials
 * embedded in both the host and a query-parameter URL) would leak.
 * Mirrors `services.llm_provider._redact_userinfo`, which uses `re.sub`
 * (all matches) for the same reason.
 */
export function redactUserinfo(url: string): string {
  return url.replace(/:\/\/[^/\s]*@/g, "://***@");
}

export type ProvidersFileResult =
  | { path: string; status: "absent" }
  | { path: string; status: "error"; error: string }
  | { path: string; status: "ok"; providers: Record<string, unknown> };

/**
 * Read + parse `config/providers.json` (or `DOKKAI_PROVIDERS_FILE`),
 * checking JSON syntax and top-level shape ONLY — the same two failure
 * modes the roadmap calls out as boot-fatal (malformed JSON, a missing/
 * mistyped `providers` key). Never inspects individual entries (see the
 * module docstring's governing principle) — an entry's `api`/`baseUrl`/
 * `apiKey` fields, id charset, or collisions with a built-in are the
 * API's own loader's job to validate, not this module's to restate.
 *
 * The path is resolved relative to `home` (mirroring `./dev.sh`'s `cd` to
 * the repo root before the API reads the same relative default) — a
 * containerized API (`docker compose --profile full`) resolves it inside
 * the container instead, which this can't see; that's a known limitation
 * of running diagnostics from outside the container.
 */
export function loadProvidersFile(home: string | undefined): ProvidersFileResult {
  const override = envVar("DOKKAI_PROVIDERS_FILE", "").trim();
  const base = home ?? process.cwd();
  const path = override
    ? isAbsolute(override)
      ? override
      : join(base, override)
    : join(base, "config", "providers.json");

  if (!existsSync(path)) {
    return { path, status: "absent" };
  }

  let isFile: boolean;
  try {
    isFile = statSync(path).isFile();
  } catch (e) {
    return { path, status: "error", error: `cannot stat: ${e instanceof Error ? e.message : String(e)}` };
  }
  if (!isFile) {
    return {
      path,
      status: "error",
      error: "expected a JSON file, found something else (a directory?)",
    };
  }

  let raw: string;
  try {
    raw = readFileSync(path, "utf8");
  } catch (e) {
    return { path, status: "error", error: `cannot read: ${e instanceof Error ? e.message : String(e)}` };
  }

  let data: unknown;
  try {
    data = JSON.parse(raw);
  } catch {
    // Deliberately NOT `e.message`, and no self-computed position either:
    // V8's own SyntaxError message can embed a snippet of the file's
    // source (which, for this file, can be a key) — the leak this guards
    // against. A hand-rolled JSON-position locator would close that leak
    // too, but it's a second JSON parser this module would have to keep
    // correct forever, and being wrong about a position in exactly the
    // command someone runs when things are broken is its own version of
    // the confidently-wrong diagnostic this module exists to avoid.
    // `./dev.sh`/`docker compose logs` (this file is loaded at API boot)
    // already reports the exact line/column/char via
    // `services.llm_provider`'s own `json.JSONDecodeError` handling — the
    // API is the authority on that, matching every other case in this
    // module where a claim beyond "here's what I can see" is left to it
    // (see `checkProvidersFile`'s wrapping message, which points there
    // for every `"error"` variant, not just this one).
    return { path, status: "error", error: "is not valid JSON" };
  }

  if (typeof data !== "object" || data === null || Array.isArray(data)) {
    return { path, status: "error", error: "top-level JSON must be an object with a 'providers' key" };
  }
  const providers = (data as Record<string, unknown>)["providers"];
  if (providers === undefined) {
    return { path, status: "error", error: "top-level JSON is missing the required 'providers' key" };
  }
  if (typeof providers !== "object" || providers === null || Array.isArray(providers)) {
    return { path, status: "error", error: "'providers' must be a JSON object of {id: entry}" };
  }
  return { path, status: "ok", providers: providers as Record<string, unknown> };
}

/**
 * Findings for `config/providers.json` itself — always relevant
 * regardless of `DESC_PROVIDER`: a malformed file breaks the whole API's
 * boot (it's loaded once at import time by `services.llm_provider`), not
 * just the descriptor. Empty when the file is absent AND
 * `DOKKAI_PROVIDERS_FILE` wasn't explicitly set — that's the ordinary,
 * unconfigured case and not worth a finding.
 *
 * The "info" case reports entry count and ids as a NEUTRAL OBSERVATION —
 * not a validation pass, deliberately "info" rather than "ok": an id that
 * would actually fail to boot (charset, shadows a built-in, duplicates
 * another after normalization, ...) still shows up here as just an id;
 * this function only speaks to whether the FILE parses, never to whether
 * any entry in it is well-formed, so it never claims "ok" about a file
 * that might not actually boot the API. Ids are admin-authored JSON keys
 * and — the realistic mistake this exists to guard against — can
 * themselves be a credential-bearing string pasted into the wrong slot
 * (e.g. a gateway URL meant for `baseUrl`); redacted the same way a base
 * URL is, before ever being joined into a message.
 */
export function checkProvidersFile(file: ProvidersFileResult): Finding[] {
  if (file.status === "absent") {
    if (!envVar("DOKKAI_PROVIDERS_FILE", "").trim()) return [];
    return [finding("warning", `DOKKAI_PROVIDERS_FILE is set but ${file.path} does not exist — no-op`)];
  }
  if (file.status === "error") {
    return [
      finding(
        "warning",
        `${file.path}: ${file.error} — this file is loaded once at API boot, so the API will fail ` +
          "to start until it's fixed; run `./dev.sh` (or `docker compose logs`) for the exact error, " +
          "e.g. the line and column of a JSON syntax mistake",
      ),
    ];
  }
  const ids = Object.keys(file.providers).map(redactUserinfo);
  if (ids.length === 0) {
    return [finding("info", `${file.path}: no providers registered`)];
  }
  return [
    finding(
      "info",
      `${file.path}: ${ids.length} provider(s) registered (${ids.join(", ")}) — ` +
        "the API validates each entry at boot; this only confirms the file parses",
    ),
  ];
}

/**
 * The JSON entry keyed by `descProvider` (already normalized: lowercased,
 * trimmed), normalized the same way the API's loader normalizes ids
 * (`.strip().lower()`) before comparing — a legal mixed-case id like
 * `"Groq"` in the file must match `DESC_PROVIDER=groq`, not be reported
 * unreachable over a casing difference this module fails to account for.
 * `undefined` if no key matches.
 */
function findEntry(providers: Record<string, unknown>, descProvider: string): unknown {
  const key = Object.keys(providers).find((id) => id.trim().toLowerCase() === descProvider);
  return key === undefined ? undefined : providers[key];
}

// The exact syntax `config/providers.json`'s loader resolves an `apiKey`
// reference through (`services.llm_provider._ENV_VAR_RE`) — UPPERCASE-only,
// anchored end-to-end. Used ONLY to recognize this one shape below, never
// to validate/reject anything else about the entry.
const _ENV_VAR_REF_RE = /^\$\{([A-Z_][A-Z0-9_]*)\}$/;

/**
 * Classifies an env var as `"unset"`, `"whitespace-only"`, or `"set"` —
 * deliberately WITHOUT the `.trim()`-before-truthiness-check this
 * module's other env-var reads use. `get_provider` reads a built-in's own
 * key env var, and a file entry's referenced `${VAR}`, via plain
 * `os.getenv(name, "")` truthiness — NOT `.strip()`-ed — so a
 * whitespace-only value is TRUTHY there: no missing-key error, and the
 * whitespace is sent as the literal credential. Reporting "is not set"
 * for that case would be false. (`DESC_API_KEY` is the one exception:
 * `describe.py`'s own docstring documents it as explicitly
 * `.strip()`-ed — blank/whitespace-only counts as unset FOR THAT ONE VAR
 * — so its own presence check, `descApiKeySet` below, correctly keeps
 * trimming; this function is for the two callers that must NOT.)
 */
function envVarPresence(name: string): "unset" | "whitespace-only" | "set" {
  const raw = envVar(name, "");
  if (raw === "") return "unset";
  return raw.trim() === "" ? "whitespace-only" : "set";
}

/**
 * A single, deliberately ASYMMETRIC check: if a file entry's `apiKey` is a
 * whole-string `${UPPERCASE_VAR}` reference, report the state of that
 * variable — the one narrow, fully backable claim this module still makes
 * about an entry's field, true regardless of whether the rest of the
 * entry is otherwise valid:
 *
 * - the variable is UNSET: warn, naming it — the single most likely
 *   reason a correctly-written file still fails. The remedy clause is
 *   deliberately soft ("satisfies THIS requirement") rather than a
 *   promise the API will boot — the same entry can still be boot-fatal
 *   for a reason this module doesn't check (an unknown `api` shape, id
 *   collision, ...), and telling the user "export it" as if that were
 *   the whole fix would be exactly the confidently-wrong diagnostic this
 *   module exists to avoid.
 * - the variable is SET BUT WHITESPACE-ONLY: warn, differently — this is
 *   NOT "unset" (Python's own truthiness check treats it as a real,
 *   present value and sends it as-is), so saying "is not set" would
 *   itself be false; say what's actually true instead.
 * - the variable IS set (non-whitespace): say NOTHING — a value exists,
 *   but this module cannot claim it's a VALID key, only that a value
 *   exists — silence is the honest answer, not a claim of "configured".
 * - `apiKey` is a literal, the wrong type, `$VAR` (missing braces),
 *   `${lowercase}`, or absent entirely: this module cannot tell a literal
 *   key from a syntax error the loader will reject — silence, not a
 *   guess in either direction.
 *
 * This asymmetry — warn on a definite, observable negative; stay quiet
 * everywhere else — is what keeps this module from ever being confidently
 * wrong about a field it doesn't fully validate.
 */
function checkFileApiKeyReference(entry: unknown, filePath: string, descProvider: string): Finding | undefined {
  if (typeof entry !== "object" || entry === null) return undefined;
  const apiKeyRaw = (entry as Record<string, unknown>)["apiKey"];
  if (typeof apiKeyRaw !== "string") return undefined;
  const match = _ENV_VAR_REF_RE.exec(apiKeyRaw.trim());
  if (!match) return undefined;
  const varName = match[1];
  const presence = envVarPresence(varName);
  if (presence === "set") return undefined;
  if (presence === "whitespace-only") {
    return finding(
      "warning",
      `${filePath}: provider '${descProvider}' apiKey references \${${varName}}, which is set but ` +
        "contains only whitespace — the API will send it as the literal credential value",
    );
  }
  return finding(
    "warning",
    `${filePath}: provider '${descProvider}' apiKey references \${${varName}}, but ${varName} is not ` +
      "set in the environment. Exporting it (or setting DESC_API_KEY) satisfies this specific " +
      "requirement — the entry may still fail to boot for a different reason this CLI doesn't check.",
  );
}

/**
 * DESC_MODEL presence — required regardless of provider, no hardcoded
 * default EVER (project rule `9a-3-clar`; `services.describe.
 * ensure_descriptor_available` hard-errors on an unset `DESC_MODEL`
 * rather than assuming one). Shared by the Ollama branch (each command's
 * own `checkOllama`) and `checkDescriptorRemote` so both phrase the same
 * fact identically.
 */
export function checkDescModel(descProvider: string): Finding {
  const descModel = envVar("DESC_MODEL", "").trim();
  if (descModel) {
    return finding("ok", `DESC_MODEL '${descModel}' configured for DESC_PROVIDER=${descProvider}`);
  }
  if (isOllamaDescriptor(descProvider)) {
    return finding(
      "warning",
      "no descriptor model specified — set DESC_MODEL (e.g. qwen2.5-coder:3b) " +
        "or configure a remote provider via DESC_PROVIDER",
    );
  }
  return finding(
    "warning",
    "DESC_MODEL is not set — description generation will fail loudly until " +
      `it's set to a model id '${descProvider}' understands`,
  );
}

/**
 * Findings for a non-Ollama `DESC_PROVIDER`. Reports what this CLI can
 * actually observe — never a claim about whether the descriptor WILL
 * work (that's the API's job, at boot/request time). Two distinct
 * "positive" levels are used deliberately:
 *
 * - "ok" for a claim this module can fully back AND that's genuinely a
 *   positive signal — an env var (`DESC_API_KEY`, a built-in's own key
 *   var) is non-empty.
 * - "info" for a neutral observation this module makes no judgment on —
 *   an id exists as a JSON key in `config/providers.json`, a
 *   `DESC_BASE_URL` value is present. Neither is validated: a JSON entry
 *   might still fail every other check the API's own loader runs
 *   (charset, shadowing, its own `api`/`baseUrl`/`apiKey` shape — none of
 *   that is re-checked here), and a base URL might be missing its scheme
 *   (`localhost:1234/v1`) — the single likeliest real mistake in exactly
 *   this configuration. Porting `services.llm_provider.validate_base_url`
 *   to TypeScript to catch that would be the same drift surface as the
 *   browser-side URL validator deleted in C5; instead this reports the
 *   value (redacted) as what it is — present — and leaves validity to
 *   the API.
 */
export function checkDescriptorRemote(descProvider: string, file: ProvidersFileResult): Finding[] {
  const findings: Finding[] = [checkDescModel(descProvider)];

  const descBaseUrl = envVar("DESC_BASE_URL", "").trim();
  const descApiKeySet = envVar("DESC_API_KEY", "").trim().length > 0;

  const builtin = BUILTIN_DESCRIPTOR_PROVIDERS.get(descProvider);
  if (builtin) {
    if (descApiKeySet) {
      findings.push(finding("ok", `${builtin.displayName} API key present (DESC_API_KEY)`));
    } else {
      // Not `.trim()`-gated the way `descApiKeySet` above is — see
      // `envVarPresence`'s docstring: `get_provider` reads a built-in's
      // key env var via plain (non-`.strip()`-ed) truthiness, so a
      // whitespace-only value is a real, sent-as-is credential in
      // Python, not a missing one.
      const presence = envVarPresence(builtin.keyEnv);
      if (presence === "set") {
        findings.push(finding("ok", `${builtin.displayName} API key present (${builtin.keyEnv})`));
      } else if (presence === "whitespace-only") {
        findings.push(
          finding(
            "warning",
            `${builtin.keyEnv} is set but contains only whitespace — the API will send it as the ` +
              `literal credential value; set DESC_API_KEY or a real ${builtin.keyEnv} instead`,
          ),
        );
      } else {
        findings.push(
          finding("warning", `${builtin.displayName} requires an API key — set DESC_API_KEY or ${builtin.keyEnv}`),
        );
      }
    }
    if (descBaseUrl) {
      findings.push(
        finding(
          "warning",
          `DESC_BASE_URL is set (${redactUserinfo(descBaseUrl)}) — overrides ` +
            `${builtin.displayName}'s own base URL, redirecting its key to that host too; ` +
            "unset it if that wasn't intentional",
        ),
      );
    }
    return findings;
  }

  const fileEntry = file.status === "ok" ? findEntry(file.providers, descProvider) : undefined;
  if (fileEntry !== undefined) {
    findings.push(
      finding(
        "info",
        `an entry for '${descProvider}' exists in ${file.path} — the API resolves ` +
          "and validates it (shape, base URL, key) at boot/use time; this only confirms the id is present",
      ),
    );
    if (descApiKeySet) {
      // DESC_API_KEY always wins over the file's own apiKey (see
      // services.describe's key-precedence docstring) — a fully backed,
      // genuinely positive claim regardless of what the file's apiKey
      // field contains.
      findings.push(finding("ok", `${descProvider} API key configured (DESC_API_KEY)`));
    } else {
      // DESC_API_KEY is unset, so the file's OWN apiKey is what actually
      // gets used — the one narrow, fully-backable claim this module
      // still makes about an entry's field (see checkFileApiKeyReference's
      // docstring for the asymmetry: warns on a definite negative, silent
      // otherwise — never claims "configured").
      const keyWarning = checkFileApiKeyReference(fileEntry, file.path, descProvider);
      if (keyWarning) findings.push(keyWarning);
    }
    if (descBaseUrl) {
      findings.push(
        finding(
          "warning",
          `DESC_BASE_URL is set (${redactUserinfo(descBaseUrl)}) — if '${descProvider}' already has ` +
            `its own baseUrl in ${file.path}, this overrides it and redirects its key to that host too; ` +
            "unset it if that wasn't intentional",
        ),
      );
    }
    return findings;
  }

  if (descBaseUrl) {
    findings.push(
      finding(
        "info",
        `DESC_PROVIDER '${descProvider}' isn't a built-in this CLI recognizes (openai/gemini/anthropic) ` +
          `or an id found in ${file.path}. DESC_BASE_URL is set (${redactUserinfo(descBaseUrl)}) — ` +
          "dokkai will try it as a custom OpenAI-compatible endpoint there; this only confirms the " +
          "value is present, the API validates the URL and connects to it at request time",
      ),
    );
    if (!descApiKeySet) {
      findings.push(
        finding(
          "warning",
          `DESC_API_KEY is not set — '${descProvider}' will be called with no key ` +
            "(fine for a keyless local server, e.g. LM Studio/vLLM)",
        ),
      );
    }
  } else {
    findings.push(
      finding(
        "warning",
        `DESC_PROVIDER='${descProvider}' isn't a built-in this CLI recognizes (openai/gemini/anthropic) ` +
          `or an id found in ${file.path}, and DESC_BASE_URL is not set. If it's a newer built-in this ` +
          "CLI's list doesn't know about yet, or the file wasn't readable from here, this CLI's view may " +
          "be stale — the API's own error at boot/use time is authoritative.",
      ),
    );
  }
  return findings;
}
