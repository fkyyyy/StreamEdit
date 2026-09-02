#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VARIANT=a exec "$SCRIPT_DIR/run_911_factorized_identity.sh"
