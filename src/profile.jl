using TOML
using Random

# Particle-state file tag.
const PARTICLE_STATE_MAGIC = UInt8[0x4e, 0x42, 0x53, 0x31]

struct FMMProfile
    expansion_order::Int
    multipole_acceptance::Float64
    leaf_size::Int
end

struct SimulationProfile
    interaction_strength::Float64
    smoothing::Float64
    particle_radius::Float64
    barnes_hut_opening_angle::Float64
    fmm::FMMProfile
    timestep::Float64
    num_steps::Int
    verification_enabled::Bool
    verification_energy_max_particles::Int
    video_encoding_enabled::Bool
    store_data::Bool
    fps::Float64
    positions::Matrix{Float64}
    velocities::Matrix{Float64}
    masses::Vector{Float64}
end

# Require a config value.
function _required(table, key, section)
    haskey(table, key) || error("Profile is missing [$section].$key")
    return table[key]
end

# Convert 3D rows to the internal 3×N matrix layout.
function _particle_matrix(value, name)
    rows = [Float64.(row) for row in value]
    isempty(rows) && error("Profile particle field '$name' cannot be empty")
    all(length(row) == 3 for row in rows) || error("Profile particle field '$name' must contain 3D rows")
    return permutedims(reduce(vcat, (permutedims(row) for row in rows)))
end

# Generate a central-mass system with orbiting satellites.
function _generated_particles(particles, interaction_strength)
    count = Int(_required(particles, "count", "particles"))
    count > 1 || error("Profile particle count must be at least 2 for a central body system")
    seed = Int(get(particles, "seed", 1))
    rng = MersenneTwister(seed)
    total_mass = Float64(get(particles, "total_mass", 1.0e15))
    radius_min = Float64(get(particles, "radius_min", 2.0e4))
    radius_max = Float64(get(particles, "radius_max", 8.0e4))
    z_half_width = Float64(get(particles, "z_half_width", 1.0e3))

    positions = zeros(3, count)
    velocities = zeros(3, count)
    masses = zeros(count)
    
    # 1. Central Body
    central_mass = total_mass / 2.0
    masses[1] = central_mass
    positions[:, 1] .= (0.0, 0.0, 0.0)
    velocities[:, 1] .= (0.0, 0.0, 0.0)
    
    # 2. Orbiting Bodies
    orbiter_mass = central_mass / (count - 1)
    
    for index in 2:count
        masses[index] = orbiter_mass
        angle = 2π * rand(rng)
        radius = radius_min + (radius_max - radius_min) * rand(rng)
        
        positions[:, index] .= (radius * cos(angle), radius * sin(angle),
                                (2rand(rng) - 1) * z_half_width)
        
        # Calculate speed based on the central mass for a stable Keplerian orbit
        speed = sqrt(interaction_strength * central_mass / radius)
        velocities[:, index] .= (-speed * sin(angle), speed * cos(angle), 0.0)
    end
    
    return positions, velocities, masses
end

# Persist a particle state for later reloads.
function write_particle_state(path::AbstractString, positions::Matrix{Float64},
                              velocities::Matrix{Float64}, masses::Vector{Float64})
    size(positions, 1) == 3 || error("positions must have shape 3 × N")
    size(velocities) == size(positions) || error("positions and velocities must have the same shape")
    length(masses) == size(positions, 2) || error("masses must contain one value per particle")
    all(masses .> 0) || error("Particle masses must be positive")

    open(path, "w") do io
        write(io, PARTICLE_STATE_MAGIC)
        write(io, UInt64(size(positions, 2)))
        write(io, positions)
        write(io, velocities)
        write(io, masses)
    end
    return path
end

function _load_particle_state(path::AbstractString)
    open(path, "r") do io
        read(io, length(PARTICLE_STATE_MAGIC)) == PARTICLE_STATE_MAGIC ||
            error("Invalid particle state file '$path'")
        count = Int(read(io, UInt64))
        count > 0 || error("Particle state file must contain at least one particle")

        positions = Matrix{Float64}(undef, 3, count)
        velocities = Matrix{Float64}(undef, 3, count)
        masses = Vector{Float64}(undef, count)
        read!(io, positions)
        read!(io, velocities)
        read!(io, masses)
        return positions, velocities, masses
    end
end

# Load the profile and resolve the particle source.
function load_profile(path::AbstractString)
    data = TOML.parsefile(path)
    simulation = get(data, "simulation", Dict{String, Any}())
    barnes_hut = get(data, "barnes_hut", Dict{String, Any}())
    fmm = get(data, "fmm", Dict{String, Any}())
    particles = get(data, "particles", Dict{String, Any}())
    interaction_strength = Float64(get(simulation, "interaction_strength",
                                      get(simulation, "gravitational_constant", 6.67430e-11)))
    smoothing = Float64(get(simulation, "smoothing", 100.0))
    particle_radius = Float64(get(simulation, "particle_radius", 0.0))
    opening_angle = Float64(get(barnes_hut, "opening_angle", get(simulation, "opening_angle", 0.5)))
    expansion_order = Int(get(fmm, "expansion_order", 5))
    multipole_acceptance = Float64(get(fmm, "multipole_acceptance", 0.4))
    leaf_size = Int(get(fmm, "leaf_size", 20))
    timestep = Float64(get(simulation, "timestep", 1.0))
    num_steps = Int(get(simulation, "steps", 120))
    verification_enabled = Bool(get(simulation, "verification_enabled", true))
    verification_energy_max_particles = Int(get(simulation, "verification_energy_max_particles", 2000))
    video_encoding_enabled = Bool(get(simulation, "video_encoding_enabled", true))
    store_data = Bool(get(simulation, "store_data", true))
    fps = Float64(get(simulation, "fps", 1.0))

    interaction_strength > 0 || error("Profile interaction_strength must be positive")
    smoothing >= 0 || error("Profile smoothing must be non-negative")
    particle_radius >= 0 || error("Profile particle_radius must be non-negative")
    opening_angle > 0 || error("Profile opening_angle must be positive")
    expansion_order > 0 || error("Profile fmm.expansion_order must be positive")
    0 < multipole_acceptance <= 1 || error("Profile fmm.multipole_acceptance must be in (0, 1]")
    leaf_size > 0 || error("Profile fmm.leaf_size must be positive")
    timestep > 0 || error("Profile timestep must be positive")
    num_steps >= 0 || error("Profile steps must be non-negative")
    verification_energy_max_particles >= 0 || error("Profile verification_energy_max_particles must be non-negative")
    fps > 0 || error("Profile fps must be positive")

    if haskey(particles, "state_file")
        state_path = String(particles["state_file"])
        state_path = isabspath(state_path) ? state_path : joinpath(dirname(abspath(path)), state_path)
        positions, velocities, masses = _load_particle_state(state_path)
    elseif haskey(particles, "positions")
        positions = _particle_matrix(particles["positions"], "positions")
        velocities = _particle_matrix(_required(particles, "velocities", "particles"), "velocities")
        masses = Float64.(_required(particles, "masses", "particles"))
        size(positions) == size(velocities) || error("positions and velocities must have the same shape")
        length(masses) == size(positions, 2) || error("masses must contain one value per particle")
    else
        positions, velocities, masses = _generated_particles(particles, interaction_strength)
    end

    all(masses .> 0) || error("Profile masses must be positive")
    
    return SimulationProfile(interaction_strength, smoothing, particle_radius, opening_angle,
                             FMMProfile(expansion_order, multipole_acceptance, leaf_size),
                             timestep, num_steps, verification_enabled,
                             verification_energy_max_particles, video_encoding_enabled, 
                             store_data, fps,
                             positions, velocities, masses)
end
