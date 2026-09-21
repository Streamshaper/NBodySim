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
gravitational_constant = 6.67430e-11
smoothing = 100.0
particle_radius = 0.0
opening_angle = 0.5
timestep = 1.0
steps = 120

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

`opening_angle` is the Barnes-Hut opening parameter. `smoothing` is the
Plummer softening length in Barnes-Hut and FMM calculations. `particle_radius`
is the physical radius supplied to FMM for its particle geometry; it is
independent of the force softening length.