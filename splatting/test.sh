#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -eq 0 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: bash splatting/test.sh RUN_DIR [run_skyanchor options]

Train Gaussians from a completed SkyAnchor reconstruction (30k aerial + 30k ground + 40k joint).
Example: bash splatting/test.sh outputs/my_scene --output outputs/my_scene_gs
Use ./run_skyanchor.sh --help for all options. Paths are relative to the current directory.
EOF
  if [[ $# -eq 0 ]]; then exit 2; fi
  exit 0
fi

RUN_DIR="$1"
shift
exec "$ROOT/run_skyanchor.sh" --splatting-only "$RUN_DIR" "$@"
