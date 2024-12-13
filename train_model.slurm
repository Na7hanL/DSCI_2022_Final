#!/bin/bash
#SBATCH --job-name=train_basic_model
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=gpu-a100 # Name of the partition
#SBATCH --mem=10G # Amount of CPU memory requird per node
#SBATCH --gres=gpu:a100_1g.10gb:1 # Requests one Nvidia A100 GPU with 10 GB of memory.
#SBATCH --account=class-dsci2022 # Account number
#SBATCH --time=1:00:00 # Maximum wall time
#SBATCH --output=output-model%j.out # Output file (stdout), %j is job allocation number
#SBATCH --error=err-model%j.out # Error file (stderr), %j is job allocation number

# Display SLURM job information
echo "Running on $(hostname)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Task ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM Node List: $SLURM_NODELIST"
echo "SLURM CPUS on Node: $SLURM_CPUS_ON_NODE"
# Run the Python script

srun python3 ptolemy_model.py
