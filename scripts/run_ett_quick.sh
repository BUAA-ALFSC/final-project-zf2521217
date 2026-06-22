#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

bash scripts/download_ett.sh
python -m src.train --ett_quick
python -m src.evaluate --ett_quick --device_target CPU
