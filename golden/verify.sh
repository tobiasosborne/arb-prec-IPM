#!/usr/bin/env bash
# golden/verify.sh — recompute each SDP's optimal value independently and
# check that it matches provenance.json to the stated digits.
# Uses wolframscript; requires Mathematica kernel to be available.
#
# Usage: bash golden/verify.sh   (from the repo root)
# Or:    cd golden && bash verify.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "arbsdp golden-master verification (bead b09)"
echo "Running: wolframscript -file $SCRIPT_DIR/gen/verify.wl"
echo ""

wolframscript -file "$SCRIPT_DIR/gen/verify.wl"
