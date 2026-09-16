#!/usr/bin/env bash
# Run on the development Docker host as the same non-root user as Galaxy jobs.
set -euo pipefail
image="${1:?Usage: bash tests/smoke-nonroot.sh IMAGE}"
test "$(id -u)" -ne 0
docker run --rm --network none --user "$(id -u):$(id -g)" \
  --entrypoint python3 "$image" -c '
from pathlib import Path
import parameters
import run
for name in ("lookup.csv", "model_europe_v1", "model_global_v1"):
    with (Path("/app") / name).open("rb") as handle:
        assert handle.read(1), name
print("PASS: DetailView imports and model assets are readable as the Galaxy job user")
'
