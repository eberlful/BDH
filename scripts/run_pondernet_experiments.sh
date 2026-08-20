#!/usr/bin/env bash

# Run the deterministic Sudoku overfit/decoding diagnostic suite sequentially.
# The script is intentionally conservative for a 16 GB Mac Mini:
# - Byte tokenizer by default to avoid GPT-2-logit memory blow-ups.
# - FP32 by default for numerical stability on MPS.
# - one experiment at a time.
# - one log file and one status record per experiment.
# - a failed experiment does not silently stop the whole overnight batch.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${PROJECT_ROOT}/configs/bdh_cq_sudoku_byte.yaml"
RUN_ROOT="${PROJECT_ROOT}/runs/sudoku-overfit-$(date '+%Y%m%d-%H%M%S')"
LOG_DIR="${RUN_ROOT}/logs"
STATUS_FILE="${RUN_ROOT}/status.tsv"
LOCK_DIR="${PROJECT_ROOT}/.sudoku-overfit.lock"
LAST_EXIT_CODE=0

# Keep MPS-compatible operations on the CPU when PyTorch has no MPS kernel.
export PYTHONUNBUFFERED=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

mkdir -p "${RUN_ROOT}" "${LOG_DIR}"

if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "ERROR: Ein anderer PonderNet-Runner scheint bereits zu laufen: ${LOCK_DIR}" >&2
  exit 1
fi

CAFFEINATE_PID=""
cleanup() {
  if [[ -n "${CAFFEINATE_PID}" ]]; then
    kill "${CAFFEINATE_PID}" 2>/dev/null || true
  fi
  rmdir "${LOCK_DIR}" 2>/dev/null || true
}
trap cleanup EXIT

on_interrupt() {
  echo "ABORTED\t$(date '+%Y-%m-%dT%H:%M:%S%z')\tRunner wurde unterbrochen" >> "${STATUS_FILE}"
  exit 130
}
trap on_interrupt INT TERM

if command -v caffeinate >/dev/null 2>&1; then
  # Prevent macOS sleep while this shell process exists.
  caffeinate -dimsu -w "$$" >/dev/null 2>&1 &
  CAFFEINATE_PID=$!
fi

cat > "${STATUS_FILE}" <<'EOF'
status	timestamp	experiment	exit_code	log
EOF

echo "Sudoku-Multi-Seed-Generalisation startet: ${RUN_ROOT}"
echo "Projekt: ${PROJECT_ROOT}"
echo "Logs:    ${LOG_DIR}"

if [[ ! -f "${PROJECT_ROOT}/main.py" || ! -f "${CONFIG}" ]]; then
  echo "ERROR: main.py oder Konfiguration nicht gefunden." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv wurde nicht gefunden." >&2
  exit 1
fi

cd "${PROJECT_ROOT}"

TRAINING_PAIR="$(uv run python -c 'from src.data.sudoku_cot import _generate_sudoku_cot_raw_samples; s=_generate_sudoku_cot_raw_samples(4, 45, 42)[0]; print("".join(map(str, s["puzzle"])), "".join(map(str, s["solution"])))')"
read -r TRAIN_PUZZLE TRAIN_SOLUTION <<< "${TRAINING_PAIR}"
printf 'puzzle=%s\nsolution=%s\n' "${TRAIN_PUZZLE}" "${TRAIN_SOLUTION}" > "${RUN_ROOT}/training-example.txt"

latest_run_dir() {
  find "${RUN_ROOT}/training-runs" -mindepth 1 -maxdepth 1 -type d -print 2>/dev/null | sort | tail -n 1
}

run_experiment() {
  local name="$1"
  shift
  local log_file="${LOG_DIR}/${name}.log"
  local started_at
  local finished_at
  local exit_code

  started_at="$(date '+%Y-%m-%dT%H:%M:%S%z')"
  echo
  echo "================================================================"
  echo "START ${name} (${started_at})"
  echo "Log: ${log_file}"
  echo "================================================================"

  {
    echo "# Experiment: ${name}"
    echo "# Start: ${started_at}"
    echo "# Command: uv run python main.py train ${CONFIG} $*"
    echo
  } > "${log_file}"

  set +e
  uv run python main.py train "${CONFIG}" "$@" \
    --set "runs_dir=${RUN_ROOT}/training-runs" 2>&1 | tee -a "${log_file}"
  exit_code="${PIPESTATUS[0]}"
  set -e

  finished_at="$(date '+%Y-%m-%dT%H:%M:%S%z')"
  if [[ "${exit_code}" -eq 0 ]]; then
    echo -e "PASS\t${finished_at}\t${name}\t0\t${log_file}" >> "${STATUS_FILE}"
    echo "PASS ${name}"
    LAST_EXIT_CODE=0
  else
    echo -e "FAIL\t${finished_at}\t${name}\t${exit_code}\t${log_file}" >> "${STATUS_FILE}"
    echo "FAIL ${name} (exit ${exit_code}); nächstes Experiment wird trotzdem gestartet." >&2
    LAST_EXIT_CODE="${exit_code}"
  fi
  # Failures are recorded and the suite continues; the final exit code is
  # determined from status.tsv after all eligible experiments finish.
  return 0
}

run_generation_check() {
  local name="$1"
  local run_dir="$2"
  local log_file="${LOG_DIR}/${name}.log"
  local started_at
  local finished_at
  local exit_code

  started_at="$(date '+%Y-%m-%dT%H:%M:%S%z')"
  {
    echo "# Decoding check on deterministic first training example"
    echo "# Puzzle: ${TRAIN_PUZZLE}"
    echo "# Ground truth: ${TRAIN_SOLUTION}"
    echo "# Run directory: ${run_dir}"
    echo
  } > "${log_file}"

  set +e
  uv run python main.py generate "${run_dir}" "${TRAIN_PUZZLE}" --max-tokens 180 2>&1 | tee -a "${log_file}"
  exit_code="${PIPESTATUS[0]}"
  set -e
  finished_at="$(date '+%Y-%m-%dT%H:%M:%S%z')"

  if [[ "${exit_code}" -eq 0 ]]; then
    echo -e "PASS\t${finished_at}\t${name}\t0\t${log_file}" >> "${STATUS_FILE}"
    LAST_EXIT_CODE=0
  else
    echo -e "FAIL\t${finished_at}\t${name}\t${exit_code}\t${log_file}" >> "${STATUS_FILE}"
    echo "Decoding-Check ${name} fehlgeschlagen." >&2
    LAST_EXIT_CODE="${exit_code}"
  fi
  return 0
}

run_teacher_forcing_check() {
  local name="$1"
  local run_dir="$2"
  local checkpoint_name="${3:-best.pt}"
  local max_samples="${4:-4}"
  local log_file="${LOG_DIR}/${name}.log"
  local finished_at
  local exit_code

  {
    echo "# Teacher-forced versus autoregressive diagnostic"
    echo "# Run directory: ${run_dir}"
    echo "# Checkpoint: ${checkpoint_name}"
    echo "# Max samples: ${max_samples}"
    echo
  } > "${log_file}"

  set +e
  uv run python scripts/diagnose_sudoku_teacher_forcing.py "${run_dir}" \
    --checkpoint "${checkpoint_name}" \
    --max-samples "${max_samples}" 2>&1 | tee -a "${log_file}"
  exit_code="${PIPESTATUS[0]}"
  set -e
  finished_at="$(date '+%Y-%m-%dT%H:%M:%S%z')"

  if [[ "${exit_code}" -eq 0 ]]; then
    echo -e "PASS\t${finished_at}\t${name}\t0\t${log_file}" >> "${STATUS_FILE}"
    LAST_EXIT_CODE=0
  else
    echo -e "FAIL\t${finished_at}\t${name}\t${exit_code}\t${log_file}" >> "${STATUS_FILE}"
    echo "Teacher-Forcing-Check ${name} fehlgeschlagen." >&2
    LAST_EXIT_CODE="${exit_code}"
  fi
  return 0
}

run_validate() {
  local log_file="${LOG_DIR}/config-validation.log"
  set +e
  uv run python main.py validate "${CONFIG}" 2>&1 | tee "${log_file}"
  local exit_code="${PIPESTATUS[0]}"
  set -e
  if [[ "${exit_code}" -ne 0 ]]; then
    echo "ERROR: Basiskonfiguration ist ungültig. Abbruch." >&2
    exit "${exit_code}"
  fi
}

run_unit_tests() {
  local log_file="${LOG_DIR}/unit-tests.log"
  local started_at
  local finished_at
  local exit_code

  started_at="$(date '+%Y-%m-%dT%H:%M:%S%z')"
  set +e
  {
    echo "# Sudoku dataset and validator tests"
    uv run python -m unittest tests/test_sudoku_cot.py -v
    first_code="$?"
    uv run python -m unittest tests/test_sudoku_validator.py -v
    second_code="$?"
    uv run python -m unittest tests/test_pondernet.py -v
    third_code="$?"
    if [[ "${first_code}" -ne 0 ]]; then
      exit "${first_code}"
    fi
    if [[ "${second_code}" -ne 0 ]]; then
      exit "${second_code}"
    fi
    exit "${third_code}"
  } 2>&1 | tee "${log_file}"
  exit_code="${PIPESTATUS[0]}"
  set -e
  finished_at="$(date '+%Y-%m-%dT%H:%M:%S%z')"

  if [[ "${exit_code}" -ne 0 ]]; then
    echo -e "FAIL\t${finished_at}\tunit-tests\t${exit_code}\t${log_file}" >> "${STATUS_FILE}"
    echo "Unit-Tests fehlgeschlagen; Trainingsläufe werden nicht gestartet." >&2
    exit "${exit_code}"
  fi
  echo -e "PASS\t${finished_at}\tunit-tests\t0\t${log_file}" >> "${STATUS_FILE}"
}

run_unit_tests
run_validate

# Focused long-run generalization comparison. Seed 42 is retained so this
# run can be compared directly with the earlier 5,000-step result.
SEEDS=(42)
GENERALIZATION_COMMON=(
  --set model.params.n_layer=3
  --set model.params.n_embd=128
  --set model.params.learning_rate=0.0003
  --set data.params.num_samples=256
  --set data.params.validation_fraction=0.125
  --set data.params.clues=45
  --set data.params.val_clues=45
  --set data.params.batch_size=1
  --set data.params.shuffle=false
  --set trainer.max_steps=10000
  --set trainer.max_epochs=46
  --set trainer.validate_every_n_epochs=2
  --set validator.params.num_eval_samples=32
)

for seed in "${SEEDS[@]}"; do
  # Fixed-compute R=2 baseline.
  run_experiment "00-fixed-r2-seed-${seed}" \
    "${GENERALIZATION_COMMON[@]}" \
    --set "seed=${seed}" \
    --set model.params.enable_pondernet=false \
    --set model.params.latent_reasoning_steps=2
  if [[ "${LAST_EXIT_CODE}" -eq 0 ]]; then
    FIXED_RUN_DIR="$(latest_run_dir)"
    run_teacher_forcing_check "00-fixed-r2-seed-${seed}-best-teacher-forcing" "${FIXED_RUN_DIR}" "best.pt" 4
    run_teacher_forcing_check "00-fixed-r2-seed-${seed}-final-teacher-forcing" "${FIXED_RUN_DIR}" "last.pt" 4
  fi

  # Selected PonderNet candidate from the regularization sweep.
  run_experiment "01-ponder-r4-lambda04-seed-${seed}" \
    "${GENERALIZATION_COMMON[@]}" \
    --set "seed=${seed}" \
    --set model.params.enable_pondernet=true \
    --set model.params.latent_reasoning_steps=4 \
    --set model.params.ponder_lambda_p=0.40 \
    --set model.params.ponder_beta=0.001
  if [[ "${LAST_EXIT_CODE}" -eq 0 ]]; then
    PONDER_LAMBDA04_RUN_DIR="$(latest_run_dir)"
    run_teacher_forcing_check "01-ponder-r4-lambda04-seed-${seed}-best-teacher-forcing" "${PONDER_LAMBDA04_RUN_DIR}" "best.pt" 4
    run_teacher_forcing_check "01-ponder-r4-lambda04-seed-${seed}-final-teacher-forcing" "${PONDER_LAMBDA04_RUN_DIR}" "last.pt" 4
  fi
done

echo
echo "Multi-Seed-Generalisation abgeschlossen. Zusammenfassung:"
column -t -s $'\t' "${STATUS_FILE}" 2>/dev/null || sed -n '1,200p' "${STATUS_FILE}"

if grep -q '^FAIL' "${STATUS_FILE}"; then
  echo "Mindestens ein Experiment ist fehlgeschlagen. Siehe ${STATUS_FILE}." >&2
  exit 1
fi

echo "Alle geplanten Overfit-Läufe erfolgreich beendet."
echo "Die Ergebnisse jetzt in der Experimentdokumentation eintragen."
