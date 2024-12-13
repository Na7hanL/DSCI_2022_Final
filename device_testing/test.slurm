#!/bin/bash
#SBATCH --job-name=model_alexnet
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=ptolemy # Name of the partition
#SBATCH --mem=20G # Amount of CPU memory requird per node
#SBATCH --account=class-dsci2022 # Account number
#SBATCH --time=1:00:00 # Maximum wall time
#SBATCH --output=output-%j.out # Output file (stdout), %j is job allocation number
#SBATCH --error=err-%j.out # Error file (stderr), %j is job allocation number
# Display SLURM job information
echo "Running on $(hostname)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Task ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM Node List: $SLURM_NODELIST"
echo "SLURM CPUS on Node: $SLURM_CPUS_ON_NODE"


# Run the Python script
srun python3 devices.py
