#!/usr/bin/env bash
# Local CI verification script for SpikeTrack, mirroring GitHub Actions CI gates.
#
# Stages:
#   0. Sibling checkout verification (../ojas, ../tessl, ../sparsl)
#   1. Formatting verification (cargo fmt workspace packages)
#   2. No-Python verification (scripts/no_python_check.sh)
#   3. Workspace clippy check (cargo clippy --workspace --all-targets -- -D warnings)
#   4. Workspace unit tests (cargo test --workspace --release --lib)
#
# Usage:
#   ./scripts/ci_local.sh [OPTIONS]
#
# Options:
#   -h, --help       Show this help message
#   --skip-fmt       Skip formatting check stage
#   --fix-fmt        Apply cargo fmt fixes to workspace packages
#   --skip-python    Skip no-python proof stage
#   --skip-clippy    Skip clippy linting stage
#   --skip-tests     Skip workspace release tests stage

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

START_TIME=$(date +%s)
SKIPS=()

show_help() {
  cat <<'EOF'
ci_local.sh: Local pre-push CI gate for SpikeTrack

Usage:
  ./scripts/ci_local.sh [OPTIONS]

Options:
  -h, --help       Show this help message
  --skip-fmt       Skip formatting check stage
  --fix-fmt        Apply cargo fmt to workspace packages
  --skip-python    Skip no-python proof stage
  --skip-clippy    Skip clippy linting stage
  --skip-tests     Skip workspace release tests stage
EOF
}

SKIP_FMT=0
FIX_FMT=0
SKIP_PYTHON=0
SKIP_CLIPPY=0
SKIP_TESTS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      show_help
      exit 0
      ;;
    --skip-fmt)
      SKIP_FMT=1
      shift
      ;;
    --fix-fmt)
      FIX_FMT=1
      shift
      ;;
    --skip-python)
      SKIP_PYTHON=1
      shift
      ;;
    --skip-clippy)
      SKIP_CLIPPY=1
      shift
      ;;
    --skip-tests)
      SKIP_TESTS=1
      shift
      ;;
    -*)
      echo "Unknown option: $1" >&2
      show_help >&2
      exit 1
      ;;
    *)
      echo "Unexpected argument: $1" >&2
      show_help >&2
      exit 1
      ;;
  esac
done

echo "======================================================================"
echo "==> [SpikeTrack CI:Local] Starting pre-push verification gates"
echo "======================================================================"

# Stage 0: Sibling checkouts and path dependencies
echo "==> [Stage 0/4] Verifying sibling path dependencies..."
SIBLINGS=(
  "$ROOT/../ojas"
  "$ROOT/../tessl"
  "$ROOT/../sparsl"
)
for sib in "${SIBLINGS[@]}"; do
  if [[ ! -d "$sib" ]]; then
    echo "ERROR: Required sibling directory '$sib' not found!" >&2
    echo "SpikeTrack requires sibling checkouts: ../ojas, ../tessl, and ../sparsl." >&2
    exit 1
  fi
done
echo "    -> Sibling checkouts verified (ojas, tessl, sparsl)."

# Workspace packages list
WORKSPACE_PKGS=()
if command -v jq >/dev/null 2>&1; then
  while IFS= read -r name; do
    [[ -n "$name" ]] && WORKSPACE_PKGS+=(-p "$name")
  done < <(cargo metadata --no-deps --format-version 1 2>/dev/null | jq -r '.packages[].name')
fi
if [[ "${#WORKSPACE_PKGS[@]}" -eq 0 ]]; then
  WORKSPACE_PKGS=(-p st-model -p st-gate -p st-weights -p st-video -p st-eval -p st-coreml -p spiketrack)
fi

# Stage 1: Code formatting verification
if [[ "$SKIP_FMT" -eq 1 ]]; then
  echo "==> [Stage 1/4] Code formatting check skipped (--skip-fmt)"
  SKIPS+=("fmt")
elif [[ "$FIX_FMT" -eq 1 ]]; then
  echo "==> [Stage 1/4] Applying cargo fmt to workspace packages..."
  cargo fmt "${WORKSPACE_PKGS[@]}"
  echo "    -> Formatting applied."
else
  echo "==> [Stage 1/4] Code formatting verification (cargo fmt)..."
  if ! cargo fmt "${WORKSPACE_PKGS[@]}" --check; then
    echo "ERROR: Code formatting violations found. Run './scripts/ci_local.sh --fix-fmt' or 'cargo fmt' to fix." >&2
    exit 1
  fi
  echo "    -> Formatting verified clean."
fi

# Stage 2: House language policy verification (Rust only, no Python)
if [[ "$SKIP_PYTHON" -eq 1 ]]; then
  echo "==> [Stage 2/4] No-Python proof skipped (--skip-python)"
  SKIPS+=("no-python")
else
  echo "==> [Stage 2/4] Executing no-Python verification (scripts/no_python_check.sh)..."
  if [[ ! -x "$ROOT/target/release/spiketrack" ]]; then
    echo "    Building target/release/spiketrack for verification..."
    cargo build --release -p spiketrack
  fi
  ./scripts/no_python_check.sh "$ROOT/target/release/spiketrack"
  echo "    -> No-Python proof passed."
fi

# Stage 3: Strict workspace clippy check
if [[ "$SKIP_CLIPPY" -eq 1 ]]; then
  echo "==> [Stage 3/4] Workspace clippy skipped (--skip-clippy)"
  SKIPS+=("clippy")
else
  echo "==> [Stage 3/4] Workspace clippy check (cargo clippy -D warnings)..."
  cargo clippy --workspace --all-targets -- -D warnings
  echo "    -> Clippy passed cleanly."
fi

# Stage 4: Workspace tests in release mode
if [[ "$SKIP_TESTS" -eq 1 ]]; then
  echo "==> [Stage 4/4] Workspace tests skipped (--skip-tests)"
  SKIPS+=("tests")
else
  echo "==> [Stage 4/4] Workspace release unit tests (cargo test --workspace --release --lib)..."
  cargo test --workspace --release --lib --no-fail-fast -- --test-threads=1
  echo "    -> Workspace tests passed."
fi

TOTAL_TIME=$(($(date +%s) - START_TIME))
echo "======================================================================"
if [[ "${#SKIPS[@]}" -gt 0 ]]; then
  joined=$(IFS=, ; echo "${SKIPS[*]}")
  echo "==> [SpikeTrack CI:Local] OK WITH SKIPS (${joined}) in ${TOTAL_TIME}s"
else
  echo "==> [SpikeTrack CI:Local] ALL GATES PASSED in ${TOTAL_TIME}s"
fi
echo "======================================================================"
