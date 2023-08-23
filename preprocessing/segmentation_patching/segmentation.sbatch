#!/bin/bash
#SBATCH --job-name=GTEX-prep
#SBATCH --partition=cpuq
#SBATCH --array=0
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --output=./logs/%a.log
#SBATCH --mem=120000M
'#SBATCH --gres=gpu:1'


echo '----------------------------'
echo ' JOB ID: '$SLURM_ARRAY_JOB_ID
echo ' CURRENT TASK ID: '$SLURM_JOB_ID
echo ' CURRENT TASK NUMBER: '$SLURM_ARRAY_TASK_ID
echo '----------------------------'
echo ' MIN TASK ID: '$SLURM_ARRAY_TASK_MIN
echo ' MAX TASK ID: '$SLURM_ARRAY_TASK_MAX
echo ' TOTAL NUMBER OF TASKS: '$SLURM_ARRAY_TASK_COUNT
echo '----------------------------'

eval "$(conda shell.bash hook)"

conda  activate /project/alfredo/conda_envs/wsi_preprocessing
python3 segmentation.py  -index=$SLURM_ARRAY_TASK_ID -num_tasks=$SLURM_ARRAY_TASK_COUNT