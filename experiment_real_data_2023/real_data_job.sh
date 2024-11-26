#!/bin/bash
#SBATCH --job-name=experiment_real_data
#SBATCH --nodes=1
#SBATCH --mincpus=1
#SBATCH --time=0-20:00:00
#SBATCH --array=0-11

set -euo pipefail

source ../.venv/bin/activate

env_file=.env
if [ -f "$env_file" ]; then
    export $(cat "$env_file" | xargs)
    echo "loaded $env_file"
fi

python real_data_job.py $SLURM_ARRAY_TASK_ID