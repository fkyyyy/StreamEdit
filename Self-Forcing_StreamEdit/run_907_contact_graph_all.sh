#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$SCRIPT_DIR/experiments/907_contact_graph"

for config_name in no_graph distance_only shuffled source_qk; do
  bash "$SCRIPT_DIR/run_907_contact_graph.sh" \
    "$CONFIG_DIR/$config_name.env"
done
