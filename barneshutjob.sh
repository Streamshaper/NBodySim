#!/bin/bash
#SBATCH --job-name=julia_job       # Name of your job
#SBATCH --output=logs/julia_%j.out       # Output log (%j will be replaced by Job ID)
#SBATCH --error=logs/julia_%j.err        # Error log
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks=1                  # Number of tasks
#SBATCH --cpus-per-task=8           # Number of CPU cores for multithreading
#SBATCH --time=00:20:00             # Maximum run time (HH:MM:SS)
#SBATCH --partition=batch           # Specify partition/queue

# Clear previous modules and load Julia 1.11.3
module purge
module load gcc/14.2.0
module load julia/1.11.3

# Set threads matching allocated CPUs
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run your Julia script
julia --project=~/Julia/NBodySim/ -t $SLURM_CPUS_PER_TASK barneshut.jl
