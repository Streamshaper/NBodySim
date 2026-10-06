# NBodySim

NBodySim is a high-performance, Julia-based 3D particle interaction simulation framework. It supports gravitational and Coulomb interactions using multiple algorithmic solvers and is designed for both local execution and distributed computing clusters via MPI and multi-threading.

## Core Features

* **Direct Summation**: A baseline $O(N^2)$ solver for exact pairwise interaction calculations.


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
julia --project=. src/nbodysim.jl profiles/default.toml
julia --project=. src/nbodysim.jl profiles/default.toml barneshut 50
julia --project=. src/nbodysim.jl profiles/default.toml fmm
julia --project=. src/nbodysim.jl profiles/coulomb.toml
```

The optional solver name overrides `[solver].type` in the profile. To use the
configured solver, omit that argument. For compatibility, the previous
solver-first form is still accepted:

```bash
julia --project=. src/nbodysim.jl fmm profiles/default.toml
```

The individual solver entry points still work when launched directly, but the wrapper is the recommended front-end.

### Running on a SLURM Cluster

To dispatch a simulation job to a SLURM queue, use the provided `submit.sh`
bash script. It accepts a profile name and an optional solver override. The
profile defaults to `planetary` when omitted.

```bash
# Usage: sbatch scripts/submit.sh <profile_name> [solver_override]
sbatch scripts/submit.sh default fmm

```

To submit a benchmark sweep, list each desired `nodes x threads` combination.
Each pair becomes a separate Slurm job; the optional arguments after `--` are
passed to `submit.sh` as its profile and solver override:

```bash
bash scripts/benchmark.sh 1x1 1x2 2x2 2x4 -- default fmm
```

This example submits four jobs, all using the `default` profile and `fmm`
solver. The sweep script creates `logs/` if it does not already exist, and
assigns each job a name containing its node and thread counts.

You can monitor the live `.out` log of a queued job before SLURM archives it by running the included watcher script:

```bash
./watch.sh

```

## Kernels and solver support

The interaction kernel defines the force law; the solver determines how the
interactions are computed. The default kernel is Plummer-softened gravity.
Kernel settings can be placed in a `[kernel]` table. This project keeps a clear
boundary between exact direct-summation kernels and approximate tree-based
solvers: the direct solver can evaluate the full force law exactly, while the
Barnes–Hut and FMM paths remain restricted to the kernels they natively support.

```toml
[kernel]
type = "plummer_gravity"
interaction_strength = 6.67430e-11
smoothing = 100.0
```

For compatibility with existing profiles, `interaction_strength` and
`smoothing` may still be set in `[simulation]`. Explicit values in `[kernel]`
take precedence. Yukawa gravity is currently available only on the direct
solver, where the screened force is evaluated exactly:

```toml
[kernel]
type = "yukawa_gravity"
interaction_strength = 6.67430e-11
smoothing = 100.0
screening_length = 1.0e6
```

Barnes–Hut supports `plummer_gravity` and `coulomb` through a charge-aware
cell approximation built on the AHRB tree. This approximation uses a
charge-weighted monopole term plus a dipole correction to better capture
asymmetric charge distributions in far-field Coulomb interactions. It is a
pragmatic higher-order approximation for the Coulomb kernel, but it is not yet a
fully native Coulomb multipole implementation with all higher-order moments.
FMM remains native for `plummer_gravity` only, while Yukawa is intentionally
restricted to the direct solver. Coulomb interactions use masses for inertia
and charges for electric coupling:

```toml
[kernel]
type = "coulomb"
interaction_strength = 1
smoothing = 0.0

[particles]
positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
velocities = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
masses = [1.0, 1.0]
charges = [1.0, -1.0]
```

Coulomb profiles require explicit per-particle charges and an explicit initial
state; the gravity-specific generated orbit initializer does not apply.

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

* `kernel.interaction_strength`: Sets the coupling strength for the selected kernel (for example, the gravitational constant or Coulomb constant, in the chosen units).


* `kernel.smoothing`: The softening length used by the selected kernel. Barnes–Hut and FMM currently use the Plummer-gravity kernel.


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
charges = [0.0, 0.0] # Optional; omitted charges default to zero.

```

An example of a fixed state is the included `planetary.toml`, which simulates one solar-mass central body and seven smaller orbiting bodies using meters, kilograms, and a timestep of 86400.0 seconds (one day).

### Binary State Files

For simulations with hundreds of thousands of particles, storing the state in a compact binary format avoids parsing massive TOML tables and ensures reproducibility. In your profile, specify the path to the binary file:

```toml
[particles]
state_file = "particles.bin"

```

Newly written files use the NBS2 format. You can generate one directly from Julia using the
`write_particle_state` helper, which writes a format header, particle count, and
contiguous `Float64` position, velocity, mass, and charge arrays. Charges default
to zero when omitted. The reader also accepts the previous NBS1 format, assigning
zero charge to those particles:

```julia
include("src/profile.jl")
write_particle_state("particles.bin", positions, velocities, masses; charges)

```

## Outputs & Verification

Upon completion, the simulation generates the following artifacts:

* **HDF5 Data (`output/simulations/`)**: High-performance chunked and compressed data files storing particle positions and velocities at each timestep, along with static masses and charges. Kernel, solver, and integrator names are included in the metadata. Toggle this via `simulation.store_data`.


* **Video Animation (`output/`)**: If `simulation.video_encoding_enabled` is set to `true`, a dual-angle 3D MP4 animation is exported.

* **Stopwatch Log (`logs/stopwatch_v2.csv`)**: If `simulation.logging_enabled` is set to `true`, solver, kernel, integrator, resource counts, particle/step counts, simulation time, and video-encoding time are appended. This versioned file uses a new schema; existing `logs/stopwatch.csv` files are left unchanged. When video encoding is disabled, the encoding time is recorded as `0.00`.

* **Verification Logs (`logs/YYYY-MM-DD/verification_<job_id>.csv`)**: Tracks conservation metrics including center-of-mass drift, relative linear momentum change, relative angular momentum change, and relative total energy change. SLURM runs use the job ID in the filename and are stored in the same date-stamped directory used to archive SLURM job logs. Non-SLURM runs use `verification_local.csv`.



Verification can be disabled by setting `simulation.verification_enabled = false`. Total energy uses the selected kernel's pair potential and scales at $O(N^2)$. To maintain performance in large simulations, exact energy is only computed if the particle count is below `simulation.verification_energy_max_particles` (default 2000); otherwise, the energy columns return `NaN` while the $O(N)$ momentum conservation checks remain active.
