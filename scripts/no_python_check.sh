#!/bin/bash
# No-Python proof for the spiketrack binary (gp-cli-and-no-python-proof).
#   scripts/no_python_check.sh [BINARY] [-- ARGS...]
# 1. `otool -L` on the binary: every linked library must be a system library or framework
#    (/usr/lib, /System/Library); Python, libtorch, ONNX Runtime or any non-system dylib fails.
# 2. Runs BINARY ARGS (default: --help) with PATH reduced to /usr/bin:/bin minus every python
#    on it, PYTHONHOME/PYTHONPATH unset, and DYLD_* cleared, and requires exit 0.
# Writes bench/results/packaging/{otool.txt,run.txt,summary.txt}; binary size and peak memory
# (from /usr/bin/time -l) go into the summary.
set -uo pipefail
cd "$(dirname "$0")/.."
bin=${1:-target/release/spiketrack}
shift || true
[ "${1:-}" = "--" ] && shift
args=("$@")
[ ${#args[@]} -eq 0 ] && args=(--help)
out=bench/results/packaging
mkdir -p "$out"
[ -x "$bin" ] || { echo "no executable at $bin (build: tools/locked.sh build bash -c 'cargo build --release -p spiketrack')"; exit 2; }

status=0
otool -L "$bin" | tee "$out/otool.txt"
# Line 1 is the binary's own path; the rest are load commands.
bad=$(tail -n +2 "$out/otool.txt" | awk '{print $1}' | grep -Ev '^(/usr/lib/|/System/Library/)' || true)
forbidden=$(tail -n +2 "$out/otool.txt" | grep -Ei 'python|torch|onnx|libc10|mkl' || true)
if [ -n "$bad" ] || [ -n "$forbidden" ]; then
  echo "FAIL: non-system or forbidden libraries linked:"; echo "$bad"; echo "$forbidden"
  status=1
fi

# A PATH with no python on it: /usr/bin carries Apple's python3 shim, so run from a temp dir
# of symlinks to the few tools needed, rather than trusting /usr/bin.
sandbox=$(mktemp -d)
trap 'rm -rf "$sandbox"' EXIT
for t in env time; do ln -s "/usr/bin/$t" "$sandbox/$t"; done
if PATH="$sandbox" command -v python3 >/dev/null 2>&1 || PATH="$sandbox" command -v python >/dev/null 2>&1; then
  echo "FAIL: python is still reachable on the reduced PATH"; status=1
fi
abs_bin=$(cd "$(dirname "$bin")" && pwd)/$(basename "$bin")
env -i HOME="$HOME" PATH="$sandbox" SPIKETRACK_WEIGHTS="${SPIKETRACK_WEIGHTS:-}" SPIKETRACK_MODEL_DIR="${SPIKETRACK_MODEL_DIR:-}" \
  /usr/bin/time -l "$abs_bin" "${args[@]}" > "$out/run.txt" 2>&1
rc=$?
[ $rc -eq 0 ] || { echo "FAIL: $bin ${args[*]} exited $rc without python on PATH (see $out/run.txt)"; status=1; }

size=$(stat -f %z "$bin")
peak=$(awk '/maximum resident set size/ {print $1}' "$out/run.txt")
real=$(awk '/ real / {print $1}' "$out/run.txt" | head -1)
{
  echo "binary: $bin ($size bytes)"
  echo "command: ${args[*]}"
  echo "exit: $rc"
  echo "wall s: ${real:-n/a}; peak RSS bytes: ${peak:-n/a}"
  echo "linked libraries: $(($(wc -l < "$out/otool.txt") - 1)), non-system: $(echo -n "$bad" | grep -c . || true)"
  echo "verdict: $([ $status -eq 0 ] && echo PASS || echo FAIL)"
} | tee "$out/summary.txt"
exit $status
