#!/bin/sh
# E9 physical ARM/Linux benchmark entry point: no device spoofing or emulation claim.
set -eu
case "$(uname -m)" in aarch64|arm64|armv7l|armv8l) ;; *) echo 'Not a physical ARM host (or cannot attest one); E9 not executed.' >&2; exit 2 ;; esac
if [ "$#" -lt 2 ]; then echo 'Usage: edge_arm.sh MANIFEST OUTPUT_JSON [additional profile_matched flags]' >&2;exit 2;fi
MANIFEST=$1; OUTPUT=$2;shift 2
exec python -m scripts.experiments.profile_matched --all --manifest "$MANIFEST" --out "$OUTPUT" "$@"
