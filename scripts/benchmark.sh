#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 <nodes>x<threads> [...] [-- <submit.sh arguments...>]" >&2
    echo "Example: $0 1x1 1x2 2x4 -- default fmm" >&2
}

node_counts=()
thread_counts=()
while (($#)) && [[ "$1" != "--" ]]; do
    if [[ "$1" =~ ^([1-9][0-9]*)x([1-9][0-9]*)$ ]]; then
        node_counts+=("${BASH_REMATCH[1]}")
        thread_counts+=("${BASH_REMATCH[2]}")
        shift
    else
        echo "Error: Invalid benchmark configuration '$1'; expected <nodes>x<threads> with positive integers." >&2
        usage
        exit 2
    fi
done

if ((${#node_counts[@]} == 0)); then
    echo "Error: Specify at least one nodes x threads configuration." >&2
    usage
    exit 2
fi

submit_args=()
if (($#)); then
    shift
    submit_args=("$@")
fi

if ! command -v sbatch >/dev/null 2>&1; then
    echo "Error: sbatch was not found; run this script on a system with Slurm." >&2
    exit 127
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
submit_script="$script_dir/submit.sh"
cd "$repo_root"
mkdir -p logs

for i in "${!node_counts[@]}"; do
    nodes=${node_counts[$i]}
    threads=${thread_counts[$i]}
    echo "Submitting benchmark with nodes=$nodes, threads=$threads"
    sbatch \
        --job-name="NBS-n${nodes}-t${threads}" \
        --nodes="$nodes" \
        --cpus-per-task="$threads" \
        "$submit_script" \
        "${submit_args[@]}"
done
