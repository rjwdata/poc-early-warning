#!/usr/bin/env bash
# run_full_experiment.sh — orchestrates ALL 4 arms (A, B, C, D) of the
# token-reduction test end-to-end: reset -> toggle tools -> run tasks -> log.
#
# >>> THIS VERSION FIXES TWO BUGS FOUND VIA TRANSCRIPT SPOT-CHECK <<<
# (Arm C, Task 9, Repeat 1 — see the findings report addendum for full detail)
#
# BUG 1 — codegraph's hook survived "removal" and contaminated Arm C.
#   `claude mcp remove codegraph` only unregisters the MCP *tool*; whatever
#   hook `codegraph install --target=claude --yes` wires in separately was
#   NOT removed by that command and kept firing during a run that was
#   supposed to have codegraph fully off. Fix: `assert_codegraph_fully_absent`
#   below checks multiple plausible locations (MCP registration, both
#   settings.json files, the hooks directory), not just `claude mcp list`,
#   and ABORTS rather than silently continuing if anything is still found.
#
# BUG 2 — Bash commands were silently denied with no approval path.
#   `--permission-mode acceptEdits` auto-approves file edits but NOT Bash
#   commands, and headless `-p` runs have no human available to approve
#   them. A subagent's plain `ls` was denied outright, forcing an expensive
#   fallback to reading whole files one at a time. Fix: use
#   `--dangerously-skip-permissions` (confirmed flag -- turns off
#   confirmation for both file writes and shell commands for that CLI
#   invocation; the settings.json equivalent is `"defaultMode":
#   "bypassPermissions"` for a permanent default, not needed here since we
#   pass the flag on every invocation anyway). A pre-flight sanity check
#   still confirms 0 permission denials on a trivial Bash command before
#   trusting a single real task run, since flag behavior can still vary by
#   CLI version. Every real run also logs its own `permission_denials`
#   count so a recurrence is visible immediately instead of discovered
#   after the fact.
#
# >>> SECURITY NOTE <<<
# `--dangerously-skip-permissions` skips Claude Code's safety confirmations
# for command execution entirely. That's an intentional, scoped trade-off
# for this specific unattended test against a disposable/resettable repo --
# it is NOT a general-purpose recommendation. Don't reuse this flag
# casually elsewhere.
#
# >>> OUTPUT LOCATION CHANGED <<<
# This writes to a NEW directory (token-reduction-output-fixed, not the
# original token-reduction-output) so a clean re-run can't get silently
# mixed with the contaminated dataset from before these fixes.
#
# ASSUMES Section 0 (one-time setup) of the runbook is already done:
#   - repo cloned, venv set up, `make train` verified once on a clean baseline
#   - CLAUDE.md moved to CLAUDE.md.hold
#   - codegraph and rtk installed (NOT yet init'd / hooked in)
#   - Section 0.5 sanity checks passed (`codegraph --version`, `rtk --version`, `rtk gain`)
#   - the jq field-path sanity check has been run by hand once and the
#     field names below confirmed against your installed CLI
#
# >>> NOTE ON REPEATS <<<
# Defaults to REPEATS=2, REPEAT_START=1. To add repeats on top of an
# existing (fixed) results.csv without re-running earlier ones:
#     REPEATS=4 REPEAT_START=3 ./run_full_experiment.sh
#
# >>> WHAT THIS SCRIPT STILL DOES NOT DO <<<
# - It does not score pass/partial/fail — that's a judgment call against the
#   "how you'll know it's correct" column, left blank in the CSV for you.

set -euo pipefail

REPEATS="${REPEATS:-2}"
REPEAT_START="${REPEAT_START:-1}"
MODEL="claude-sonnet-5"   # must match what's pinned in runbook Step 0.2
PERMISSION_FLAG="${PERMISSION_FLAG:---dangerously-skip-permissions}"

# BUG 3 FIX: `--dangerously-skip-permissions` is refused outright when the
# CLI is run as root/sudo ("cannot be used with root/sudo privileges"), which
# is common on disposable/CI test boxes -- exactly this script's use case.
# Setting IS_SANDBOX=1 lifts that restriction. Auto-set it when running as
# root and not already set, rather than let the pre-flight check fail with
# a misleading "check claude --help for the right flag" message.
if [ "$(id -u)" -eq 0 ] && [ -z "${IS_SANDBOX:-}" ]; then
  echo "Running as root: exporting IS_SANDBOX=1 so $PERMISSION_FLAG is permitted."
  export IS_SANDBOX=1
fi

# IMPORTANT: results.csv and logs/ live OUTSIDE the repo, one directory up,
# in a NEW directory distinct from the pre-fix dataset (see note above).
REPO_ROOT="$(pwd)"
OUT_ROOT="$(dirname "$REPO_ROOT")/token-reduction-output-fixed"
LOG_DIR="$OUT_ROOT/logs"
LOG_CSV="$OUT_ROOT/results.csv"
mkdir -p "$LOG_DIR"

if [ ! -f "$LOG_CSV" ]; then
  echo "task,arm,repeat,input_tokens,output_tokens,cache_read_tokens,cache_write_tokens,cost_usd,wall_seconds,session_id,transcript_path,pass_fail,output_file,thinking_tokens,permission_denials" > "$LOG_CSV"
fi

declare -A TASKS
TASKS[1]="Map the repo structure: identify the Streamlit app entry point, the multi-page UI, the ML pipeline stages, and where trained artifacts live."
TASKS[2]="Trace the ML pipeline end-to-end: from raw data through preprocessing, training, to how a prediction actually gets served in the Streamlit app."
TASKS[3]="Inventory the data quality / model evaluation surfaces in this repo -- what gets checked, and where results are surfaced."
TASKS[4]="Stand up the local dev environment from a clean checkout and attempt to run the test suite."
TASKS[5]="Run the full training pipeline locally (make train) and confirm the expected artifacts are produced."
TASKS[6]="Run the linters (make lint) and fix any warnings they raise."
TASKS[7]="Add a first test case (create a tests/ directory if it doesn't exist) covering data_transformation.py's preprocessing pipeline."
TASKS[8]="The config/params.yaml file defines a KNNImputer configuration block (n_neighbors: 3, weights: uniform), but src/components/data_transformation.py never reads config/params.yaml at all -- it hardcodes SimpleImputer(strategy='median') instead. Fix this: wire data_transformation.py to actually read the imputer settings from config/params.yaml and use KNNImputer as configured."
TASKS[9]="Add one new predictive feature to the training pipeline (a derived column of your choice) and verify it flows through to the pages/03_predictions.py input form and prediction output."

# ---- guardrail: codegraph registered (for arms that need it ON) ----
assert_codegraph_registered() {
  local want="$1"  # "yes" or "no"
  local present="no"
  if claude mcp list 2>/dev/null | grep -qi codegraph; then present="yes"; fi
  if [ "$present" != "$want" ]; then
    echo "!!! ABORT: expected codegraph registered=$want but found registered=$present." >&2
    exit 1
  fi
}

# ---- BUG 1 FIX: best-effort full removal + thorough verification ----
remove_codegraph_completely() {
  claude mcp remove codegraph 2>/dev/null || true
  # Best-effort uninstall of whatever codegraph installed beyond the MCP
  # registration. UNVERIFIED against codegraph's actual CLI -- check
  # `codegraph --help` yourself; this is a guess at the mirror of `install`.
  codegraph uninstall --target=claude --yes 2>/dev/null || true
}

codegraph_traces_found() {
  # Returns the list of traces found (empty string = fully absent).
  # Does NOT exit -- callers decide whether that's fatal.
  local found=""
  if claude mcp list 2>/dev/null | grep -qi codegraph; then
    found="${found}MCP registration; "
  fi
  if [ -f "$HOME/.claude/settings.json" ] && grep -qi codegraph "$HOME/.claude/settings.json" 2>/dev/null; then
    found="${found}~/.claude/settings.json; "
  fi
  if [ -f "$REPO_ROOT/.claude/settings.json" ] && grep -qi codegraph "$REPO_ROOT/.claude/settings.json" 2>/dev/null; then
    found="${found}repo .claude/settings.json; "
  fi
  if ls "$HOME/.claude/hooks/" 2>/dev/null | grep -qi codegraph; then
    found="${found}~/.claude/hooks/; "
  fi
  echo -n "$found"
}

assert_codegraph_fully_absent() {
  # Hard abort -- use mid-test, before any arm that expects codegraph off.
  local found
  found="$(codegraph_traces_found)"
  if [ -n "$found" ]; then
    echo "!!! ABORT: codegraph traces still present after attempted removal: $found" >&2
    echo "!!! This is exactly the contamination found in the Arm C spot-check." >&2
    echo "!!! Fix by hand (check \`codegraph --help\` for its real uninstall command," >&2
    echo "!!! and inspect the files above directly) before continuing." >&2
    exit 1
  fi
}

# ---- BUG 2 FIX: verify Bash actually executes before trusting a real run ----
preflight_check_bash_execution() {
  echo "=== Pre-flight: confirming Bash executes non-interactively with 0 permission denials ==="
  local PF_FILE="${LOG_DIR}/preflight_bash_check.json"
  claude -p "Run the shell command: ls -la . -- then report exactly what it printed." \
    --model "$MODEL" \
    --output-format json \
    "$PERMISSION_FLAG" \
    > "$PF_FILE" 2> "${PF_FILE}.stderr" || true

  local DENIALS
  DENIALS=$(jq '.permission_denials | length' "$PF_FILE" 2>/dev/null || echo "PARSE_ERROR")

  if [ "$DENIALS" = "PARSE_ERROR" ]; then
    echo "!!! ABORT: could not parse the pre-flight check's output." >&2
    echo "!!! Inspect by hand: cat $PF_FILE | jq ." >&2
    echo "!!! This usually means $PERMISSION_FLAG isn't a valid flag on your installed" >&2
    echo "!!! CLI version -- check \`claude --help\` and set" >&2
    echo "!!! PERMISSION_FLAG=<correct flag> ./run_full_experiment.sh" >&2
    exit 1
  fi

  if [ "$DENIALS" != "0" ]; then
    echo "!!! ABORT: $DENIALS permission denial(s) on a plain 'ls' with $PERMISSION_FLAG." >&2
    echo "!!! This is the exact bug found in the Arm C spot-check -- it is NOT fixed yet." >&2
    echo "!!! Inspect: cat $PF_FILE | jq . -- check claude --help for the right non-interactive" >&2
    echo "!!! permission flag on your version before running the real test." >&2
    exit 1
  fi

  echo "Pre-flight OK: 0 permission denials on the sanity check."
}

# ---- run all 9 tasks x REPEATS for whichever arm is currently toggled on ----
run_tasks_for_arm() {
  local ARM="$1"
  for TASK_NUM in 1 2 3 4 5 6 7 8 9; do
    for REPEAT in $(seq "$REPEAT_START" "$REPEATS"); do
      echo ">>> Arm $ARM | Task $TASK_NUM | Repeat $REPEAT"

      # Section 3: reset repo state before every single run, no exceptions
      git checkout .
      git clean -fd

      OUT_FILE="${LOG_DIR}/arm${ARM}_task${TASK_NUM}_rep${REPEAT}.json"
      START=$(date +%s)

      # BUG 4 FIX: without this guard, a single non-zero exit from `claude`
      # (rate limit, network blip, transient API error) trips `set -e` and
      # kills the entire remaining experiment (all later arms/tasks/repeats)
      # with no logged row. Capture the exit code, log a warning, and let
      # the loop continue -- the row still gets written (fields will read
      # PARSE_ERROR/NA if the output file is empty) so the failure is visible
      # in results.csv instead of silently truncating the run.
      CLAUDE_EXIT=0
      claude -p "${TASKS[$TASK_NUM]}" \
        --model "$MODEL" \
        --output-format json \
        "$PERMISSION_FLAG" \
        > "$OUT_FILE" 2> "${OUT_FILE}.stderr" || CLAUDE_EXIT=$?

      if [ "$CLAUDE_EXIT" != "0" ]; then
        echo "!!! WARNING: claude exited $CLAUDE_EXIT for Arm $ARM Task $TASK_NUM Rep $REPEAT -- see ${OUT_FILE}.stderr" >&2
      fi

      END=$(date +%s)
      WALL=$((END - START))

      INPUT_TOK=$(jq '.usage.input_tokens // "NA"' "$OUT_FILE" 2>/dev/null || echo "PARSE_ERROR")
      OUTPUT_TOK=$(jq '.usage.output_tokens // "NA"' "$OUT_FILE" 2>/dev/null || echo "PARSE_ERROR")
      CACHE_READ=$(jq '.usage.cache_read_input_tokens // "NA"' "$OUT_FILE" 2>/dev/null || echo "NA")
      CACHE_WRITE=$(jq '.usage.cache_creation_input_tokens // "NA"' "$OUT_FILE" 2>/dev/null || echo "NA")
      SESSION_ID=$(jq -r '.session_id // "NA"' "$OUT_FILE" 2>/dev/null || echo "NA")
      COST_USD=$(jq '.total_cost_usd // "NA"' "$OUT_FILE" 2>/dev/null || echo "NA")
      THINKING_TOK=$(jq '.usage.output_tokens_details.thinking_tokens // "NA"' "$OUT_FILE" 2>/dev/null || echo "NA")
      PERM_DENIALS=$(jq '.permission_denials | length // "NA"' "$OUT_FILE" 2>/dev/null || echo "NA")

      TRANSCRIPT="NA"
      if [ "$SESSION_ID" != "NA" ]; then
        FOUND=$(find ~/.claude/projects -type f -name "${SESSION_ID}.jsonl" 2>/dev/null | head -n1 || true)
        [ -n "$FOUND" ] && TRANSCRIPT="$FOUND"
      fi

      echo "${TASK_NUM},${ARM},${REPEAT},${INPUT_TOK},${OUTPUT_TOK},${CACHE_READ},${CACHE_WRITE},${COST_USD},${WALL},${SESSION_ID},${TRANSCRIPT},,${OUT_FILE},${THINKING_TOK},${PERM_DENIALS}" >> "$LOG_CSV"

      if [ "$INPUT_TOK" = "PARSE_ERROR" ]; then
        echo "!!! Could not parse usage from $OUT_FILE -- inspect by hand: cat $OUT_FILE | jq ."
      fi
      if [ "$PERM_DENIALS" != "0" ] && [ "$PERM_DENIALS" != "NA" ]; then
        echo "!!! WARNING: Arm $ARM Task $TASK_NUM Rep $REPEAT had $PERM_DENIALS permission denial(s)."
        echo "!!! The Bash-approval bug may still be occurring for some tool/subagent path"
        echo "!!! that the pre-flight check didn't cover. Inspect: cat $OUT_FILE | jq ."
      fi
    done
  done
}

# ============================ Pre-flight ============================
preflight_check_bash_execution

# =========================== ARM A: Baseline ===========================
echo "=== ARM A: Baseline (codegraph off, rtk off) ==="
remove_codegraph_completely
assert_codegraph_fully_absent
if ls ~/.claude/hooks/ 2>/dev/null | grep -qi rtk; then
  echo "!!! ABORT: an rtk hook is present but Arm A expects none active." >&2
  exit 1
fi
run_tasks_for_arm A

# ======================= ARM B: Indexing only =======================
echo "=== ARM B: Indexing only (codegraph on, rtk off) ==="
codegraph install --target=claude --yes
codegraph init -i
assert_codegraph_registered "yes"
run_tasks_for_arm B

echo "--- removing codegraph before Arm C (leaving .codegraph/ index data on disk) ---"
remove_codegraph_completely
assert_codegraph_fully_absent

# =================== ARM C: Output-limiting only ===================
echo "=== ARM C: Output-limiting only (codegraph off, rtk on) ==="
rtk init --global
assert_codegraph_fully_absent
run_tasks_for_arm C

# ========================= ARM D: Combined =========================
echo "=== ARM D: Combined (codegraph on, rtk on) ==="
claude mcp add codegraph -- codegraph serve --mcp
assert_codegraph_registered "yes"
rtk gain || { echo "!!! ABORT: rtk gain failed -- rtk hook may not be active for Arm D." >&2; exit 1; }
run_tasks_for_arm D

# ============================ Cleanup ============================
echo "=== Cleanup ==="
remove_codegraph_completely
CLEANUP_TRACES="$(codegraph_traces_found)"
if [ -n "$CLEANUP_TRACES" ]; then
  echo "!!! WARNING: codegraph traces remain after final cleanup: $CLEANUP_TRACES"
  echo "!!! Clean up by hand before your next run."
fi

echo "--- disabling rtk's global hook ---"
rtk init -g --uninstall 2>/dev/null || true
if ls ~/.claude/hooks/ 2>/dev/null | grep -qi rtk; then
  echo "!!! WARNING: an rtk hook is still present after \`rtk init -g --uninstall\`."
  echo "!!! Clean up by hand (check \`rtk --help\`) before your next run."
else
  echo "rtk hook confirmed removed."
fi

echo "!!! MANUAL STEP: mv CLAUDE.md.hold CLAUDE.md, then git checkout . && git clean -fd"

echo "All arms complete. Results in $LOG_CSV. Fill in the pass_fail column by hand"
echo "against the 'how you'll know it's correct' table in the runbook (Section 1)."
echo "Check the permission_denials column too -- anything nonzero needs a look."
