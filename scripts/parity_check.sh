#!/usr/bin/env bash
# PyTorch Parity Verification Runner for SpikeTrack.
# Executes vision backbone and tracker numerical parity checks against PyTorch reference fixtures
# (visem_22_f0.safetensors, judo.safetensors) on CPU and Metal backends.
#
# Usage:
#   ./scripts/parity_check.sh [OPTIONS]
#
# Examples:
#   ./scripts/parity_check.sh                             # Runs CPU backbone parity checks (default)
#   ./scripts/parity_check.sh --backend metal             # Runs Metal GPU backbone parity checks
#   ./scripts/parity_check.sh --backend all               # Runs CPU and Metal backbone parity checks
#   ./scripts/parity_check.sh --component tracker         # Runs tracker parity checks (CPU)
#   ./scripts/parity_check.sh --component all --backend metal
#   ./scripts/parity_check.sh --test vit_layer_norm_eps   # Runs single test filter
#   ./scripts/parity_check.sh --debug                     # Runs with debug profile (default: release)
#   ./scripts/parity_check.sh --no-gpu-lock

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

START_TIME=$(date +%s)

TMP_LOG_FILES=()
cleanup_logs() {
  if [[ ${#TMP_LOG_FILES[@]} -gt 0 ]]; then
    for f in "${TMP_LOG_FILES[@]}"; do
      [[ -f "$f" ]] && rm -f "$f"
    done
  fi
  return 0
}
trap cleanup_logs EXIT INT TERM

show_help() {
  cat <<'EOF'
parity_check.sh: PyTorch Oracle Fixture Parity Verification Runner for SpikeTrack

Usage:
  ./scripts/parity_check.sh [OPTIONS]

Options:
  -b, --backend BACKEND        Target backend: cpu, metal, or all (default: cpu)
  -c, --component COMPONENT    Component: backbone, tracker, or all (default: backbone)
  -t, --test TEST_FILTER       Filter to specific exact test name (e.g. vit_layer_norm_eps_matches_upstream, multiplex_mask_decoder_matches_upstream)
  -r, --release                Run with cargo --release (default)
  -d, --debug                  Run with cargo debug profile
  --weights PATH               Explicit path to SAM 3.1 multiplex checkpoint safetensors
  --fixtures PATH              Explicit path to backbone fixtures directory
  --tracker-fixtures PATH      Explicit path to tracker fixtures safetensors
  --no-gpu-lock                Do not require or attempt mac_heavy GPU lock for Metal
  -h, --help                   Show this help message

Environment Variables:
  SPIKETRACK_WEIGHTS           Path to sam3.1_multiplex.safetensors
  SPIKETRACK_FIXTURES          Path to backbone fixtures directory
  SPIKETRACK_TRACKER_FIXTURES  Path to tracker fixtures safetensors
  MAC_HEAVY_LABEL              Indicates active exclusive GPU lock under mac_heavy.sh
EOF
}

BACKEND="cpu"
COMPONENT="backbone"
TEST_FILTER=""
PROFILE="release"
CUSTOM_WEIGHTS=""
CUSTOM_FIXTURES=""
CUSTOM_TRACKER_FIXTURES=""
NO_GPU_LOCK=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      show_help
      exit 0
      ;;
    -b|--backend)
      BACKEND="${2:-}"
      shift 2
      ;;
    -c|--component)
      COMPONENT="${2:-}"
      shift 2
      ;;
    -t|--test)
      TEST_FILTER="${2:-}"
      shift 2
      ;;
    -r|--release)
      PROFILE="release"
      shift
      ;;
    -d|--debug)
      PROFILE="debug"
      shift
      ;;
    --weights)
      CUSTOM_WEIGHTS="${2:-}"
      shift 2
      ;;
    --fixtures)
      CUSTOM_FIXTURES="${2:-}"
      shift 2
      ;;
    --tracker-fixtures)
      CUSTOM_TRACKER_FIXTURES="${2:-}"
      shift 2
      ;;
    --no-gpu-lock)
      NO_GPU_LOCK=1
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      show_help >&2
      exit 1
      ;;
  esac
done

# Validate backend
case "$BACKEND" in
  cpu|metal|all) ;;
  *)
    echo "ERROR: Invalid backend '$BACKEND'. Must be 'cpu', 'metal', or 'all'." >&2
    exit 1
    ;;
esac

# Validate component
case "$COMPONENT" in
  backbone|tracker|all) ;;
  *)
    echo "ERROR: Invalid component '$COMPONENT'. Must be 'backbone', 'tracker', or 'all'." >&2
    exit 1
    ;;
esac

# Resolve Weights path
if [[ -n "$CUSTOM_WEIGHTS" ]]; then
  export SPIKETRACK_WEIGHTS="$CUSTOM_WEIGHTS"
elif [[ -z "${SPIKETRACK_WEIGHTS:-}" ]]; then
  WEIGHTS_CANDIDATES=(
    "$ROOT/models/sam3.1/sam3.1_multiplex.safetensors"
    "/Users/bharath/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
    "${HOME}/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
    "$ROOT/../models/sam3.1/sam3.1_multiplex.safetensors"
  )
  for cand in "${WEIGHTS_CANDIDATES[@]}"; do
    if [[ -f "$cand" ]]; then
      export SPIKETRACK_WEIGHTS="$cand"
      break
    fi
  done
fi

if [[ -z "${SPIKETRACK_WEIGHTS:-}" || ! -f "$SPIKETRACK_WEIGHTS" ]]; then
  echo "======================================================================" >&2
  echo "ERROR: SAM 3.1 checkpoint file not found!" >&2
  echo "Looked for: sam3.1_multiplex.safetensors" >&2
  echo "Set SPIKETRACK_WEIGHTS or pass --weights <path>." >&2
  echo "======================================================================" >&2
  exit 1
fi

# Resolve Backbone Fixtures path
if [[ -n "$CUSTOM_FIXTURES" ]]; then
  export SPIKETRACK_FIXTURES="$CUSTOM_FIXTURES"
elif [[ -z "${SPIKETRACK_FIXTURES:-}" ]]; then
  FIXTURE_CANDIDATES=(
    "$ROOT/fixtures/backbone"
    "/Users/bharath/Code/research/SpikeTrack/fixtures/backbone"
    "${HOME}/Code/research/SpikeTrack/fixtures/backbone"
  )
  for cand in "${FIXTURE_CANDIDATES[@]}"; do
    if [[ -d "$cand" && -f "$cand/visem_22_f0.safetensors" ]]; then
      export SPIKETRACK_FIXTURES="$cand"
      break
    fi
  done
fi

# Resolve Tracker Fixtures path
if [[ -n "$CUSTOM_TRACKER_FIXTURES" ]]; then
  export SPIKETRACK_TRACKER_FIXTURES="$CUSTOM_TRACKER_FIXTURES"
elif [[ -z "${SPIKETRACK_TRACKER_FIXTURES:-}" ]]; then
  TRACKER_CANDIDATES=(
    "$ROOT/fixtures/tracker/judo.safetensors"
    "$ROOT/fixtures/vos/judo.safetensors"
    "/Users/bharath/Code/research/SpikeTrack/fixtures/tracker/judo.safetensors"
    "${HOME}/Code/research/SpikeTrack/fixtures/tracker/judo.safetensors"
  )
  for cand in "${TRACKER_CANDIDATES[@]}"; do
    if [[ -f "$cand" ]]; then
      export SPIKETRACK_TRACKER_FIXTURES="$cand"
      break
    fi
  done
fi

PROFILE_FLAG=""
if [[ "$PROFILE" == "release" ]]; then
  PROFILE_FLAG="--release"
fi

echo "======================================================================"
echo "==> [Parity Check] PyTorch Oracle Parity Runner for SpikeTrack"
echo "    Backend:           $BACKEND"
echo "    Component:         $COMPONENT"
echo "    Test Filter:       ${TEST_FILTER:-<all>}"
echo "    Profile:           $PROFILE"
echo "    Weights:           $SPIKETRACK_WEIGHTS ($(du -h "$SPIKETRACK_WEIGHTS" | cut -f1))"
if [[ -n "${SPIKETRACK_FIXTURES:-}" ]]; then
  echo "    Backbone Fixtures: $SPIKETRACK_FIXTURES"
fi
if [[ -n "${SPIKETRACK_TRACKER_FIXTURES:-}" ]]; then
  echo "    Tracker Fixtures:  $SPIKETRACK_TRACKER_FIXTURES"
fi
echo "======================================================================"

run_test_with_verification() {
  local suite_label="$1"
  local test_binary="$2"
  shift 2
  local -a cmd=("$@")

  local log_file
  log_file=$(mktemp /tmp/spiketrack_parity_XXXXXX)
  TMP_LOG_FILES+=("$log_file")

  set +e
  "${cmd[@]}" 2>&1 | tee "$log_file"
  local rc=${PIPESTATUS[0]}
  set -e

  if [[ $rc -ne 0 ]]; then
    rm -f "$log_file"
    echo "ERROR: Parity test suite '$suite_label' failed with exit code $rc" >&2
    exit "$rc"
  fi

  if grep -qE "(running 0 tests|test result: .* 0 passed; 0 failed)" "$log_file"; then
    rm -f "$log_file"
    echo "" >&2
    echo "======================================================================" >&2
    if [[ -n "$TEST_FILTER" ]]; then
      echo "ERROR: No parity tests matched filter '$TEST_FILTER' in $suite_label suite!" >&2
    else
      echo "ERROR: 0 parity tests were executed in $suite_label suite!" >&2
    fi
    echo "Available parity test binary: $test_binary" >&2
    echo "======================================================================" >&2
    exit 1
  fi

  rm -f "$log_file"
  return 0
}

GPU_LOCK_CMD=()
resolve_gpu_lock() {
  GPU_LOCK_CMD=()

  if [[ -n "${MAC_HEAVY_LABEL:-}" ]]; then
    echo "    GPU Lock: Active under MAC_HEAVY_LABEL='$MAC_HEAVY_LABEL'"
    return 0
  fi

  if [[ "$NO_GPU_LOCK" -eq 1 ]]; then
    echo "    GPU Lock: Bypassed (--no-gpu-lock)"
    return 0
  fi

  local LOCK_SCRIPTS=(
    "$ROOT/../Lappi-decision/tools/mac_heavy.sh"
    "/Users/bharath/Code/research/Lappi-decision/tools/mac_heavy.sh"
    "$ROOT/../tessl/tools/mac_heavy.sh"
    "${HOME}/Code/research/Lappi-decision/tools/mac_heavy.sh"
  )
  for s in "${LOCK_SCRIPTS[@]}"; do
    if [[ -x "$s" ]]; then
      GPU_LOCK_CMD=("$s" "spiketrack-parity")
      echo "    GPU Lock: Wrapping with $s"
      return 0
    fi
  done

  echo "    Notice: mac_heavy.sh not found. Running with --test-threads=1 serialization."
  return 0
}

run_backbone_suite() {
  local target_backend="$1"
  echo ""
  echo "==> [Backbone Parity - $target_backend] Running st-model backbone_parity..."

  if [[ -z "${SPIKETRACK_FIXTURES:-}" || ! -d "$SPIKETRACK_FIXTURES" ]]; then
    echo "ERROR: Backbone fixtures directory not found!" >&2
    echo "Looked for: $SPIKETRACK_FIXTURES" >&2
    exit 1
  fi

  local filter_arg="$TEST_FILTER"
  if [[ -z "$filter_arg" ]]; then
    case "$target_backend" in
      cpu)   filter_arg="backbone_matches_pytorch_fp32_on_five_images" ;;
      metal) filter_arg="backbone_matches_pytorch_fp32_on_five_images_on_metal" ;;
    esac
  fi

  if [[ "$target_backend" == "cpu" && "$filter_arg" =~ metal ]]; then
    echo "ERROR: Metal test '$filter_arg' cannot be executed when target backend is 'cpu'." >&2
    exit 1
  fi

  local base_cmd=(cargo test $PROFILE_FLAG -p st-model --test backbone_parity)
  if [[ -n "$filter_arg" ]]; then
    base_cmd+=("$filter_arg")
  fi
  base_cmd+=(-- --exact --ignored --nocapture --test-threads=1)

  local full_cmd=()
  if [[ "$target_backend" == "metal" ]]; then
    resolve_gpu_lock
    if [[ "${#GPU_LOCK_CMD[@]}" -gt 0 ]]; then
      full_cmd=("${GPU_LOCK_CMD[@]}" "${base_cmd[@]}")
    else
      full_cmd=("${base_cmd[@]}")
    fi
  else
    full_cmd=("${base_cmd[@]}")
  fi

  echo "    Command: ${full_cmd[*]}"
  run_test_with_verification "Backbone ($target_backend)" "tests/backbone_parity.rs" "${full_cmd[@]}"
}

run_tracker_suite() {
  local target_backend="$1"
  echo ""
  echo "==> [Tracker Parity - $target_backend] Running st-model tracker_parity..."

  if [[ -z "${SPIKETRACK_TRACKER_FIXTURES:-}" || ! -f "$SPIKETRACK_TRACKER_FIXTURES" ]]; then
    echo "ERROR: Tracker fixtures file not found!" >&2
    echo "Looked for: judo.safetensors" >&2
    exit 1
  fi

  local filter_arg="$TEST_FILTER"
  if [[ -z "$filter_arg" ]]; then
    case "$target_backend" in
      cpu)   filter_arg="multiplex_mask_decoder_matches_upstream" ;;
      metal) filter_arg="multiplex_mask_decoder_matches_upstream_on_metal" ;;
    esac
  fi

  if [[ "$target_backend" == "cpu" && "$filter_arg" =~ metal ]]; then
    echo "ERROR: Metal test '$filter_arg' cannot be executed when target backend is 'cpu'." >&2
    exit 1
  fi

  local base_cmd=(cargo test $PROFILE_FLAG -p st-model --test tracker_parity)
  if [[ -n "$filter_arg" ]]; then
    base_cmd+=("$filter_arg")
  fi
  base_cmd+=(-- --exact --ignored --nocapture --test-threads=1)

  local full_cmd=()
  if [[ "$target_backend" == "metal" ]]; then
    resolve_gpu_lock
    if [[ "${#GPU_LOCK_CMD[@]}" -gt 0 ]]; then
      full_cmd=("${GPU_LOCK_CMD[@]}" "${base_cmd[@]}")
    else
      full_cmd=("${base_cmd[@]}")
    fi
  else
    full_cmd=("${base_cmd[@]}")
  fi

  echo "    Command: ${full_cmd[*]}"
  run_test_with_verification "Tracker ($target_backend)" "tests/tracker_parity.rs" "${full_cmd[@]}"
}

# Execution dispatcher
execute_suites() {
  local target_backend="$1"
  case "$COMPONENT" in
    backbone)
      run_backbone_suite "$target_backend"
      ;;
    tracker)
      run_tracker_suite "$target_backend"
      ;;
    all)
      run_backbone_suite "$target_backend"
      run_tracker_suite "$target_backend"
      ;;
  esac
}

case "$BACKEND" in
  cpu)
    execute_suites "cpu"
    ;;
  metal)
    if [[ "$(uname -s)" != "Darwin" ]]; then
      echo "ERROR: Metal backend tests require macOS Darwin (Apple Silicon)." >&2
      exit 1
    fi
    execute_suites "metal"
    ;;
  all)
    execute_suites "cpu"
    if [[ "$(uname -s)" == "Darwin" ]]; then
      execute_suites "metal"
    fi
    ;;
esac

TOTAL_TIME=$(($(date +%s) - START_TIME))
echo ""
echo "======================================================================"
echo "==> [Parity Check] SUCCESS in ${TOTAL_TIME}s"
echo "======================================================================"
