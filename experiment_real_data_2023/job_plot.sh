#!/bin/bash
#SBATCH --job-name=experiment_real_data
#SBATCH --nodes=1
#SBATCH --mincpus=1
#SBATCH --time=0-01:00:00

set -euo pipefail

source ../.venv/bin/activate

env_file=.env
if [ -f "$env_file" ]; then
    export $(cat "$env_file" | xargs)
    echo "loaded $env_file"
fi

python plot_trace_hist.py
