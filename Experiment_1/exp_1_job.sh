#!/bin/bash
#SBATCH --job-name=experiment_1_array
#SBATCH --nodes=1
#SBATCH --mincpus=1
#SBATCH --output=/dev/null
#SBATCH --array=1-50
#SBATCH --time=0-20:00:00

set -euo pipefail
 
source ../.venv/bin/activate

env_file=.env
if [ -f "$env_file" ]; then
    export $(cat "$env_file" | xargs)
    echo "loaded $env_file"
fi

python Experiment_1_monsoon.py $SLURM_ARRAY_TASK_ID