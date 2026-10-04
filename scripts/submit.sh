#!/bin/bash
#SBATCH --job-name=NBS		       	# Job name
#SBATCH --output=logs/julia_%j.out 	# Output log (%j will be replaced by Job ID)
#SBATCH --error=logs/julia_%j.err  	# Error log
#SBATCH --nodes=2                   # Number of nodes
#SBATCH --ntasks-per-node=1         # Keep ranks distributed across nodes
#SBATCH --cpus-per-task=8           # Number of Julia threads per MPI rank
#SBATCH --time=00:30:00             # Maximum run time (HH:MM:SS)
#SBATCH --partition=rome           # Specify partition/queue

FIRST_ARG=${1:-planetary}
SECOND_ARG=${2:-}
THIRD_ARG=${3:-}

is_solver() {
    case "${1,,}" in
        direct|d|barneshut|bh|fmm|f) return 0 ;;
        *) return 1 ;;
    esac
}

# Keep accepting the previous solver-first invocation.
if is_solver "$FIRST_ARG"; then
    SOLVER_OVERRIDE=$FIRST_ARG
    PROFILE_BASE=${SECOND_ARG:-planetary}
    if [ -n "$THIRD_ARG" ]; then
        echo "Error: The profile-first submission form accepts at most a profile and solver."
        echo "Usage: sbatch $0 <profile_name> [solver_override]"
        exit 1
    fi
else
    SCRIPT_BASE=${FIRST_ARG%.jl}
    if [ "$SCRIPT_BASE" = "nbodysim" ]; then
        SOLVER_OVERRIDE=$SECOND_ARG
        PROFILE_BASE=${THIRD_ARG:-planetary}
    elif [ "$SCRIPT_BASE" = "direct" ] || [ "$SCRIPT_BASE" = "barneshut" ] || [ "$SCRIPT_BASE" = "fmm" ]; then
        SOLVER_OVERRIDE=$SCRIPT_BASE
        PROFILE_BASE=${SECOND_ARG:-planetary}
    elif [[ "$FIRST_ARG" == *.jl ]]; then
        JULIA_SCRIPT="src/${SCRIPT_BASE}.jl"
        PROFILE_BASE=${SECOND_ARG:-planetary}
    else
        if [ -n "$THIRD_ARG" ]; then
            echo "Error: Expected a profile name and optional solver override."
            echo "Usage: sbatch $0 <profile_name> [solver_override]"
            exit 1
        fi
        PROFILE_BASE=$FIRST_ARG
        SOLVER_OVERRIDE=$SECOND_ARG
        SCRIPT_BASE=nbodysim
    fi
fi

if [ -n "${SOLVER_OVERRIDE:-}" ] && ! is_solver "$SOLVER_OVERRIDE"; then
    echo "Error: Unknown solver override '$SOLVER_OVERRIDE'."
    echo "Expected one of: direct, barneshut, fmm."
    exit 1
fi

PROFILE_BASE=${PROFILE_BASE%.toml}
PROFILE="profiles/${PROFILE_BASE}.toml"
if [ -z "${JULIA_SCRIPT:-}" ]; then
    JULIA_SCRIPT="src/nbodysim.jl"
fi
JULIA_ARGS=("$JULIA_SCRIPT" "$PROFILE")
if [ -n "${SOLVER_OVERRIDE:-}" ]; then
    JULIA_ARGS+=("$SOLVER_OVERRIDE")
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
if [ -n "${SLURM_JOB_ID:-}" ]; then
DATE_DIR="logs/$(date +%Y-%m-%d)"
mkdir -p "$DATE_DIR"

# Move and rename this job's log files into the date folder
mv "logs/julia_${SLURM_JOB_ID}.out" "$DATE_DIR/${SCRIPT_BASE:-nbodysim}_${PROFILE_BASE}_${SLURM_JOB_ID}.out"
mv "logs/julia_${SLURM_JOB_ID}.err" "$DATE_DIR/${SCRIPT_BASE:-nbodysim}_${PROFILE_BASE}_${SLURM_JOB_ID}.err"
fi
