# NBodySim

NBodySim is a high-performance, Julia-based 3D gravitational N-body simulation framework. It calculates gravitational interactions using multiple algorithmic solvers and is designed for both local execution and distributed computing clusters via MPI and multi-threading.

## Core Features

* **Direct Summation**: A baseline $O(N^2)$ solver for exact gravitational interaction calculations.


* **Barnes-Hut**: A tree-based approximation algorithm utilizing `AdaptiveHierarchicalRegularBinning` for spatial partitioning.


* **Fast Multipole Method (FMM)**: An advanced, highly scalable approximation solver utilizing the `FastMultipole` package.


* **Hybrid Parallelization**: Distributes the workload across cluster nodes using MPI, while utilizing Julia's native multi-threading (`Threads.@threads`) for calculations on each rank.


* **Automated Visualization**: Automatically renders and exports 3D animated visualizations of the simulation to `.mp4` using `CairoMakie` based on the configurable frames per second (`fps`).


* **Physics Verification**: Computes system energy, center-of-mass drift, and momentum/angular momentum conservation, writing checkpoint metrics to CSV logs.



## Requirements and Installation

NBodySim requires **Julia 1.11.3** and an MPI implementation (e.g., MPICH or OpenMPI).

The repository utilizes `DrWatson.jl` for environment management. If running on the **Aristotle HPC** (where this project was developed), you must export the following environment variables before downloading or installing Julia packages to ensure proper network routing and stability:

```bash
export JULIA_DOWNLOADS_USE_CURL=true
export JULIA_PKG_SERVER=""
export JULIA_NUM_THREADS=1

```

Once your environment variables are set, instantiate the project environment:

```julia
using Pkg
Pkg.instantiate()

```

## Usage

Use the front-end wrapper in the src directory to select the solver. Solver and
integrator defaults can be set in `[solver]` and `[integrator]`; an explicit
solver argument on the command line overrides the profile's solver. Solver,
kernel, and integrator compatibility is validated when loading the profile.

### Running Locally

```bash
julia --project=. src/nbodysim.jl direct profiles/default.toml
julia --project=. src/nbodysim.jl barneshut profiles/default.toml 50
julia --project=. src/nbodysim.jl fmm profiles/default.toml
```

To use the solver selected in the profile, omit the solver argument:

```bash
julia --project=. src/nbodysim.jl profiles/default.toml
```

The individual solver entry points still work when launched directly, but the wrapper is the recommended front-end.

### Running on a SLURM Cluster

To dispatch a simulation job to a SLURM queue, use the provided `submit.sh` bash script. It requires the target Julia script (the solver) and an optional profile name. It falls back to the `planetary` profile if omitted.

```bash
# Usage: sbatch scripts/submit.sh <model_name> [profile_name]
sbatch scripts/submit.sh fmm default

```

You can monitor the live `.out` log of a queued job before SLURM archives it by running the included watcher script:

```bash
./watch.sh

```

## Kernels and solver support

The interaction kernel defines the force law; the solver determines how the
interactions are computed. The default kernel is Plummer-softened gravity.
Kernel settings can be placed in a `[kernel]` table:

```toml
[kernel]
type = "plummer_gravity"
interaction_strength = 6.67430e-11
smoothing = 100.0
```

For compatibility with existing profiles, `interaction_strength` and
`smoothing` may still be set in `[simulation]`. Explicit values in `[kernel]`
take precedence. The direct solver also supports screened Yukawa gravity:

```toml
[kernel]
type = "yukawa_gravity"
interaction_strength = 6.67430e-11
smoothing = 100.0
screening_length = 1.0e6
```

Barnes–Hut and FMM currently support only `plummer_gravity`. Selecting another
kernel with either solver produces an explicit unsupported-kernel error.

The integrator is selected separately in an `[integrator]` table. The current
implementation supports semi-implicit Euler:

```toml
[integrator]
type = "semi_implicit_euler"
```

Select the solver in a `[solver]` table. The default profile selects direct
summation; Barnes–Hut and FMM keep their algorithm-specific settings in their
existing tables:

```toml
[solver]
type = "barneshut"
```

## Configuration Profiles

Simulations are controlled entirely via `.toml` configuration profiles. Profiles contain physical parameters and particle generation settings.

### Core Parameters

* `interaction_strength`: Sets the effective pairwise interaction strength (e.g., the gravitational constant for Newtonian gravity).


* `smoothing`: The Plummer softening length utilized in Barnes-Hut and FMM calculations to prevent singularities.


* `particle_radius`: The physical radius supplied to FMM for its particle geometry, independent of the force softening length.


* `barnes_hut.opening_angle`: The Barnes-Hut opening parameter ($\theta$) that determines when to approximate a distant cluster of masses as a single node.


* `fmm.expansion_order`: The FMM multipole expansion order.


* `fmm.multipole_acceptance`: The FMM multipole acceptance criterion.


* `fmm.leaf_size`: The maximum number of particles per FMM leaf.



### Particle Generation

You can define dynamic particle generation by providing a `count`, `seed`, `total_mass`, and spatial constraints (`radius_min`, `radius_max`, `z_half_width`).

Alternatively, for small, fully fixed initial states, you can provide the 3D rows directly in the TOML file:

```toml
[particles]
positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
velocities = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
masses = [1.0, 1.0]

```

An example of a fixed state is the included `planetary.toml`, which simulates one solar-mass central body and seven smaller orbiting bodies using meters, kilograms, and a timestep of 86400.0 seconds (one day).

### Binary State Files

For simulations with hundreds of thousands of particles, storing the state in a compact binary format avoids parsing massive TOML tables and ensures reproducibility. In your profile, specify the path to the binary file:

```toml
[particles]
state_file = "particles.bin"

```

You can generate this binary file directly from Julia using the `write_particle_state` helper, which writes a format header, the particle count, and the contiguous `Float64` position, velocity, and mass arrays:

```julia
include("src/profile.jl")
write_particle_state("particles.bin", positions, velocities, masses)

```

## Outputs & Verification

Upon completion, the simulation generates the following artifacts:

* **HDF5 Data (`output/simulations/`)**: High-performance chunked and compressed data files storing particle positions, velocities, and static masses at each timestep. Toggle this via `simulation.store_data`.


* **Video Animation (`output/`)**: If `simulation.video_encoding_enabled` is set to `true`, a dual-angle 3D MP4 animation is exported.

* **Stopwatch Log (`logs/stopwatch_v2.csv`)**: If `simulation.logging_enabled` is set to `true`, solver, kernel, integrator, resource counts, particle/step counts, simulation time, and video-encoding time are appended. This versioned file uses a new schema; existing `logs/stopwatch.csv` files are left unchanged. When video encoding is disabled, the encoding time is recorded as `0.00`.

* **Verification Logs (`logs/YYYY-MM-DD/verification_<job_id>.csv`)**: Tracks conservation metrics including center-of-mass drift, relative linear momentum change, relative angular momentum change, and relative total energy change. SLURM runs use the job ID in the filename and are stored in the same date-stamped directory used to archive SLURM job logs. Non-SLURM runs use `verification_local.csv`.



Verification can be disabled by setting `simulation.verification_enabled = false`. Total energy calculation uses the Plummer-softened potential and scales at $O(N^2)$. To maintain performance in large simulations, exact energy is only computed if the particle count is below `simulation.verification_energy_max_particles` (default 2000); otherwise, the energy columns return `NaN` while the $O(N)$ momentum conservation checks remain active.
