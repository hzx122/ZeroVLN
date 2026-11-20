#!/usr/bin/env bash
# Interactive v8/RLOO navigation on this GPU host; external port defaults to 5002.
set -euo pipefail
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
exec "$REPO/scripts/eval/run_evaluation.sh" console "$@"
