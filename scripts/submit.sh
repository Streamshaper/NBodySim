#!/bin/bash
#SBATCH --job-name=NBS		       	# Job name
#SBATCH --output=logs/julia_%j.out 	# Output log (%j will be replaced by Job ID)
#SBATCH --error=logs/julia_%j.err  	# Error log
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks-per-node=1         # Keep ranks distributed across nodes
#SBATCH --cpus-per-task=8           # Number of Julia threads per MPI rank
#SBATCH --time=00:20:00             # Maximum run time (HH:MM:SS)
#SBATCH --partition=batch           # Specify partition/queue

# Strip extensions if the user accidentally includes them
SCRIPT_BASE=${1%.jl}
PROFILE_BASE=${2%.toml}

# Validate that the Julia script argument was provided
if [ -z "$SCRIPT_BASE" ]; then
    echo "Error: You must provide a Julia script name."
    echo "Usage: sbatch $0 <script_name> [profile_name]"
    exit 1
fi

# Fallback to 'planetary' if no profile is provided
PROFILE_BASE=${PROFILE_BASE:-planetary}

# Automatically construct the full paths with extensions
JULIA_SCRIPT="src/${SCRIPT_BASE}.jl"
PROFILE="profiles/${PROFILE_BASE}.toml"

# Clear previous modules and load Julia 1.11.3
module purge
module load gcc/14.2.0
module load julia/1.11.3

# Use all allocated CPUs through MPI plus Julia threads
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Launch one Julia process per allocated MPI task and connect all tasks to one MPI world.
srun --mpi=pmi2 --ntasks="$SLURM_NTASKS" --ntasks-per-node=1 \
    julia --project=~/Julia/NBodySim/ -t "$SLURM_CPUS_PER_TASK" "$JULIA_SCRIPT" "$PROFILE"


# LOG ORGANIZATION
# Create a subfolder for current date
DATE_DIR="logs/$(date +%Y-%m-%d)"
mkdir -p "$DATE_DIR"

# Move and rename this job's log files into the date folder
mv "logs/julia_${SLURM_JOB_ID}.out" "$DATE_DIR/${SCRIPT_BASE}_${PROFILE_BASE}_${SLURM_JOB_ID}.out"
mv "logs/julia_${SLURM_JOB_ID}.err" "$DATE_DIR/${SCRIPT_BASE}_${PROFILE_BASE}_${SLURM_JOB_ID}.err"
