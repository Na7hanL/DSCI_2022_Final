#!/bin/bash
#SBATCH --job-name=deviceTest
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=gpu-a100 
#SBATCH --mem=10G 
#SBATCH --account=class-dsci2022 
#SBATCH --gres=gpu:a100_1g.10gb:1 
#SBATCH --time=1:00:00 
#SBATCH --output=output-%j.out 
#SBATCH --error=err-%j.out 
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=nml162@msstate.edu 

echo "Running on $(hostname)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Task ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM Node List: $SLURM_NODELIST"
echo "SLURM CPUS on Node: $SLURM_CPUS_ON_NODE"

export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

module load cuda
module load cudnn

srun python3 devices.py

