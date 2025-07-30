#!/bin/bash
#SBATCH -n 1
######SBATCH --array=0-110
#######SBATCH --array=0-80
#SBATCH --array=0-20
#SBATCH -c 5
#######BATCH --output=/ru-auth/local/home/fbuck/scratch/test_rslds_params/logs/explained_var_%A_%a.out
#######SBATCH --error=/ru-auth/local/home/fbuck/scratch/test_rslds_params/logs/explained_var_%A_%a.err

# Load modules or activate conda environment if needed

eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate sleap  # Activate conda if using it

PY_SCRIPT="/ru-auth/local/home/fbuck/scratch/test_rslds_params/rslds_utils/run_rslds_fitting_params_HPC_2.py"

subsample_id="explained_var"
rslds_type="vanilla"
save_dir_parent="/ru-auth/local/home/fbuck/scratch/test_rslds_params/"
# Run the script with the current task ID as idx
python "$PY_SCRIPT" --params_i $SLURM_ARRAY_TASK_ID  --subsample_id "$subsample_id" --rslds_type "$rslds_type" --save_dir_parent "$save_dir_parent"


rslds_type="hiearch"
# Run the script with the current task ID as idx
python "$PY_SCRIPT" --params_i $SLURM_ARRAY_TASK_ID  --subsample_id "$subsample_id" --rslds_type "$rslds_type" --save_dir_parent "$save_dir_parent"
#####SBATCH --array=0-110