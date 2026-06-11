#!/bin/sh
# Generate synthetic analysis datasets (Lorenz, Thomas, Hindmarsh-Rose).
# Output: data/analysis/{system}/noise-{level}/{params}/train|val|test_data.npy
set -eu

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

ANALYSIS="data/process/analysis"
CONFIGS="$ANALYSIS/configs"
COMMANDS="$ANALYSIS/generate_data_commands.sh"

: > "$COMMANDS"

run_configs() {
    base="$1"
    shift
    for noise in "$@"; do
        python "$ANALYSIS/generate_configs.py" -c "$CONFIGS/$base" -n "$noise"
    done
}

run_configs lorenz-base.yaml 0 1 3 5
run_configs thomas-base.yaml 0 1 3 5
run_configs rose-base.yaml 0 1 3 5

# Optional systems (uncomment to include):
# run_configs rossler-base.yaml 0 1 3 5
# run_configs vanderpols-base.yaml 5

while IFS= read -r cmd; do
    [ -z "$cmd" ] && continue
    echo "Running: $cmd"
    eval "$cmd"
done < "$COMMANDS"

python "$ANALYSIS/generate_run_commands.py" \
    -c configs/analysis-w-100 \
    -s scripts/generated/commands_analysis.sh
