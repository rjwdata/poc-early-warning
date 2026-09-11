#!/usr/bin/env bash
# run_full_experiment.sh — orchestrates ALL 4 arms (A, B, C, D) of the
# token-reduction test end-to-end: reset -> toggle tools -> run tasks -> log.
#
# ASSUMES Section 0 (one-time setup) of the runbook is already done:
#   - repo cloned, venv set up, `make train` verified once on a clean baseline
#   - CLAUDE.md moved to CLAUDE.md.hold
#   - codegraph and rtk installed (NOT yet init'd / hooked in)
#   - Section 0.5 sanity checks passed (`codegraph --version`, `rtk --version`, `rtk gain`)
#   - the jq field-path sanity check at the bottom of run_arm.sh has been run by
#     hand once and the field names below confirmed against your installed CLI
#
# >>> NOTE ON REPEATS <<<
# The runbook recommends never running a cell only once (Section 6: "cut to 2
# minimum"), since a single run can't distinguish real variance from a fluke.
# This script defaults to 2 repeats per task/arm (the runbook's stated minimum).
# Override with, e.g.:
#     REPEATS=3 ./run_full_experiment.sh
#
# >>> WHAT THIS SCRIPT DOES NOT DO <<<
# - It does not guess rtk's disable command (unverified in the runbook —
#   final cleanup prints a reminder instead of a command).
# - It does not score pass/partial/fail — that's a judgment call against the
#   "how you'll know it's correct" column, left blank in the CSV for you.
# - It does not run /cost — usage numbers come straight from the JSON output.

set -euo pipefail

REPEATS="${REPEATS:-2}"
MODEL="claude-sonnet-5"   # must match what's pinned in runbook Step 0.2
LOG_CSV="results.csv"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

if [ ! -f "$LOG_CSV" ]; then
  echo "task,arm,repeat,input_tokens,output_tokens,cache_read_tokens,cache_write_tokens,cost_usd,wall_seconds,session_id,transcript_path,pass_fail,output_file" > "$LOG_CSV"
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

# ---- guardrail: confirm codegraph's registration state matches what the arm expects ----
assert_codegraph_registered() {
  local want="$1"  # "yes" or "no"
  local present="no"
  if claude mcp list 2>/dev/null | grep -qi codegraph; then present="yes"; fi
  if [ "$present" != "$want" ]; then
    echo "!!! ABORT: expected codegraph registered=$want but found registered=$present." >&2
    echo "!!! Fix this by hand (claude mcp list / claude mcp remove codegraph) before re-running." >&2
    exit 1
  fi
}

# ---- run all 9 tasks x REPEATS for whichever arm is currently toggled on ----
run_tasks_for_arm() {
  local ARM="$1"
  for TASK_NUM in 1 2 3 4 5 6 7 8 9; do
    for REPEAT in $(seq 1 "$REPEATS"); do
      echo ">>> Arm $ARM | Task $TASK_NUM | Repeat $REPEAT"

      # Section 3: reset repo state before every single run, no exceptions
      git checkout .
      git clean -fd

      OUT_FILE="${LOG_DIR}/arm${ARM}_task${TASK_NUM}_rep${REPEAT}.json"
      START=$(date +%s)

      claude -p "${TASKS[$TASK_NUM]}" \
        --model "$MODEL" \
        --output-format json \
        --permission-mode acceptEdits \
        > "$OUT_FILE" 2> "${OUT_FILE}.stderr"

      END=$(date +%s)
      WALL=$((END - START))

      # Field paths confirmed by hand against a live `claude -p ... --output-format json`
      # sanity check on this installed CLI version -- no fallbacks needed, they matched exactly.
      INPUT_TOK=$(jq '.usage.input_tokens // "NA"' "$OUT_FILE" 2>/dev/null || echo "PARSE_ERROR")
      OUTPUT_TOK=$(jq '.usage.output_tokens // "NA"' "$OUT_FILE" 2>/dev/null || echo "PARSE_ERROR")
      CACHE_READ=$(jq '.usage.cache_read_input_tokens // "NA"' "$OUT_FILE" 2>/dev/null || echo "NA")
      CACHE_WRITE=$(jq '.usage.cache_creation_input_tokens // "NA"' "$OUT_FILE" 2>/dev/null || echo "NA")
      SESSION_ID=$(jq -r '.session_id // "NA"' "$OUT_FILE" 2>/dev/null || echo "NA")
      COST_USD=$(jq '.total_cost_usd // "NA"' "$OUT_FILE" 2>/dev/null || echo "NA")

      TRANSCRIPT="NA"
      if [ "$SESSION_ID" != "NA" ]; then
        FOUND=$(find ~/.claude/projects -type f -name "${SESSION_ID}.jsonl" 2>/dev/null | head -n1 || true)
        [ -n "$FOUND" ] && TRANSCRIPT="$FOUND"
      fi

      echo "${TASK_NUM},${ARM},${REPEAT},${INPUT_TOK},${OUTPUT_TOK},${CACHE_READ},${CACHE_WRITE},${COST_USD},${WALL},${SESSION_ID},${TRANSCRIPT},,${OUT_FILE}" >> "$LOG_CSV"

      if [ "$INPUT_TOK" = "PARSE_ERROR" ]; then
        echo "!!! Could not parse usage from $OUT_FILE -- inspect by hand: cat $OUT_FILE | jq ."
      fi
    done
  done
}

# =========================== ARM A: Baseline ===========================
echo "=== ARM A: Baseline (codegraph off, rtk off) ==="
claude mcp remove codegraph 2>/dev/null || true
assert_codegraph_registered "no"
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

echo "--- removing codegraph before Arm C (leaving .codegraph/ on disk) ---"
claude mcp remove codegraph
assert_codegraph_registered "no"

# =================== ARM C: Output-limiting only ===================
echo "=== ARM C: Output-limiting only (codegraph off, rtk on) ==="
rtk init --global
assert_codegraph_registered "no"
run_tasks_for_arm C

# ========================= ARM D: Combined =========================
echo "=== ARM D: Combined (codegraph on, rtk on) ==="
claude mcp add codegraph -- codegraph serve --mcp
assert_codegraph_registered "yes"
rtk gain || { echo "!!! ABORT: rtk gain failed -- rtk hook may not be active for Arm D." >&2; exit 1; }
run_tasks_for_arm D

# ============================ Cleanup ============================
echo "=== Cleanup ==="
claude mcp remove codegraph 2>/dev/null || true
echo "!!! MANUAL STEP: disable rtk's global hook yourself (check \`rtk --help\` for the"
echo "    exact command -- the runbook flags this as unverified, so this script won't guess)."
echo "!!! MANUAL STEP: mv CLAUDE.md.hold CLAUDE.md, then git checkout . && git clean -fd"

echo "All arms complete. Results in $LOG_CSV. Fill in the pass_fail column by hand"
echo "against the 'how you'll know it's correct' table in the runbook (Section 1)."
