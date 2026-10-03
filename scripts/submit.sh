#!/bin/bash
#SBATCH --job-name=NBS		       	# Job name
#SBATCH --output=logs/julia_%j.out 	# Output log (%j will be replaced by Job ID)
#SBATCH --error=logs/julia_%j.err  	# Error log
#SBATCH --nodes=2                   # Number of nodes
#SBATCH --ntasks-per-node=1         # Keep ranks distributed across nodes
#SBATCH --cpus-per-task=64           # Number of Julia threads per MPI rank
#SBATCH --time=02:00:00             # Maximum run time (HH:MM:SS)
#SBATCH --partition=rome           # Specify partition/queue

# Strip extensions if the user accidentally includes them
FIRST_ARG=${1:-}
SECOND_ARG=${2:-}
THIRD_ARG=${3:-}
SCRIPT_BASE=${FIRST_ARG%.jl}
PROFILE_BASE=${SECOND_ARG%.toml}

if [ "$SCRIPT_BASE" = "nbodysim" ]; then
    SCRIPT_BASE=${SECOND_ARG:-}
    PROFILE_BASE=${THIRD_ARG%.toml}
fi

# Validate that the Julia script argument was provided
if [ -z "$SCRIPT_BASE" ]; then
    echo "Error: You must provide a model or Julia script name."
    echo "Usage: sbatch $0 <direct|barneshut|fmm|script_name> [profile_name]"
    exit 1
fi

# Fallback to 'planetary' if no profile is provided
PROFILE_BASE=${PROFILE_BASE:-planetary}

# Use the front-end wrapper when a solver model is chosen directly.
if [ "$SCRIPT_BASE" = "direct" ] || [ "$SCRIPT_BASE" = "barneshut" ] || [ "$SCRIPT_BASE" = "fmm" ]; then
    JULIA_SCRIPT="src/nbodysim.jl"
    PROFILE="profiles/${PROFILE_BASE}.toml"
    JULIA_ARGS=("$JULIA_SCRIPT" "$SCRIPT_BASE" "$PROFILE")
else
    # Automatically construct the full paths with extensions for direct solver entry points.
    JULIA_SCRIPT="src/${SCRIPT_BASE}.jl"
    PROFILE="profiles/${PROFILE_BASE}.toml"
    JULIA_ARGS=("$JULIA_SCRIPT" "$PROFILE")
fi

# Clear previous modules and load Julia 1.11.3
module purge
module load gcc/14.2.0
module load julia/1.11.3

# Use all allocated CPUs through MPI plus Julia threads
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Launch one Julia process per allocated MPI task and connect all tasks to one MPI world.
srun --mpi=pmi2 --ntasks="$SLURM_NTASKS" --ntasks-per-node=1 \
    julia --project=~/Julia/NBodySim/ -t "$SLURM_CPUS_PER_TASK" "${JULIA_ARGS[@]}"


# LOG ORGANIZATION
# Create a subfolder for current date
DATE_DIR="logs/$(date +%Y-%m-%d)"
mkdir -p "$DATE_DIR"

# Move and rename this job's log files into the date folder
mv "logs/julia_${SLURM_JOB_ID}.out" "$DATE_DIR/${SCRIPT_BASE}_${PROFILE_BASE}_${SLURM_JOB_ID}.out"
mv "logs/julia_${SLURM_JOB_ID}.err" "$DATE_DIR/${SCRIPT_BASE}_${PROFILE_BASE}_${SLURM_JOB_ID}.err"
