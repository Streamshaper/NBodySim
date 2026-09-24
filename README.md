# NBodySim profiles

Each solver accepts a TOML profile as its first argument:

```text
julia --project=. src/fmm.jl profiles/default.toml
julia --project=. src/barneshut.jl profiles/default.toml 50
```

The optional second argument overrides the number of steps. A profile contains
the physical parameters and either a reproducible disk generator or an exact
particle state:
```toml
[simulation]
interaction_strength = 6.67430e-11
smoothing = 100.0
particle_radius = 0.0
timestep = 1.0
steps = 120

[barnes_hut]
opening_angle = 0.5

[fmm]
expansion_order = 5
multipole_acceptance = 0.4
leaf_size = 20

[particles]
count = 1000
seed = 42
total_mass = 1.0e15
radius_min = 2.0e4
radius_max = 8.0e4
z_half_width = 1.0e3
```

For a small fully fixed initial state, matching three-dimensional rows can be
stored directly in TOML:

```toml
[particles]
positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
velocities = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
masses = [1.0, 1.0]
```

Each solver writes checkpoint verification metrics to `logs/verification_*.csv` and
prints the final values. The metrics include center-of-mass drift, relative linear
momentum change, relative angular-momentum change, and relative total-energy change.
Set `simulation.verification_enabled = false` to disable all verification work and
CSV output while retaining the energy limit for enabled runs.
The energy uses the same Plummer-softened potential as the force calculation and is
computed exactly only when the particle count is at most
`simulation.verification_energy_max_particles` (default `2000`); otherwise the energy
columns are `NaN` while the O(N) conservation checks remain active.

Set `simulation.video_encoding_enabled = false` to skip frame retention and CairoMakie
video encoding entirely.
Set `simulation.fps` to control the encoded video's frame rate; it defaults to `1.0`.

For hundreds of thousands of particles, store the state in the compact binary
format used by `profiles/fixed_state.toml`:

```toml
[particles]
state_file = "particles.bin"
```

Create the file from Julia with `write_particle_state`:

```julia
include("src/profile.jl")
write_particle_state("particles.bin", positions, velocities, masses)
```

The binary file contains a small format header, the particle count, then the
contiguous `Float64` position, velocity, and mass arrays. This avoids parsing
large TOML tables and keeps the exact initial state reproducible.

`barnes_hut.opening_angle` is the Barnes-Hut opening parameter.
`fmm.expansion_order` is the FMM multipole expansion order,
`fmm.multipole_acceptance` is the FMM multipole acceptance criterion, and
`fmm.leaf_size` is the number of particles per FMM leaf. `interaction_strength`
sets the effective pairwise interaction strength for the selected physical model
(for Newtonian gravity this is the gravitational constant). `smoothing` is the
Plummer softening length in Barnes-Hut and FMM calculations. `particle_radius`
is the physical radius supplied to FMM for its particle geometry; it is
independent of the force softening length.

`profiles/planetary.toml` contains a hardcoded 11-body system: one solar-mass
central body and ten smaller orbiting bodies. Its positions use metres, velocities
use metres per second, masses use kilograms, and its timestep is one day. Video
encoding is enabled by default for this profile.