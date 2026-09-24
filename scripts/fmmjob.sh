#!/bin/bash
#SBATCH --job-name=julia_job       # Name of your job
#SBATCH --output=logs/julia_%j.out       # Output log (%j will be replaced by Job ID)
#SBATCH --error=logs/julia_%j.err        # Error log
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks-per-node=1         # Keep ranks distributed across nodes
#SBATCH --cpus-per-task=8           # Number of Julia threads per MPI rank
#SBATCH --time=00:20:00             # Maximum run time (HH:MM:SS)
#SBATCH --partition=batch           # Specify partition/queue

# Clear previous modules and load Julia 1.11.3
module purge
module load gcc/14.2.0
module load julia/1.11.3

# Use all allocated CPUs through MPI plus Julia threads
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
PROFILE=${PROFILE:-profiles/planetary.toml}

# Launch one Julia process per allocated MPI task and connect all tasks to one MPI world.
	srun --mpi=pmi2 --ntasks="$SLURM_NTASKS" --ntasks-per-node=1 \
		julia --project=~/Julia/NBodySim/ -t "$SLURM_CPUS_PER_TASK" src/fmm.jl "$PROFILE"
