#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VARIANT=b exec "$SCRIPT_DIR/run_911_factorized_identity.sh"
