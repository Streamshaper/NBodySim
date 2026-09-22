using TOML
using Random

const PARTICLE_STATE_MAGIC = UInt8[0x4e, 0x42, 0x53, 0x31]

struct SimulationProfile
    gravitational_constant::Float64
    smoothing::Float64
    particle_radius::Float64
    opening_angle::Float64
    timestep::Float64
    num_steps::Int
    verification_enabled::Bool
    verification_energy_max_particles::Int
    video_encoding_enabled::Bool
    fps::Float64
    positions::Matrix{Float64}
    velocities::Matrix{Float64}
    masses::Vector{Float64}
end

function _required(table, key, section)
    haskey(table, key) || error("Profile is missing [$section].$key")
    return table[key]
end

function _particle_matrix(value, name)
    rows = [Float64.(row) for row in value]
    isempty(rows) && error("Profile particle field '$name' cannot be empty")
    all(length(row) == 3 for row in rows) || error("Profile particle field '$name' must contain 3D rows")
    return permutedims(reduce(vcat, (permutedims(row) for row in rows)))
end

function _generated_particles(particles, gravitational_constant)
    count = Int(_required(particles, "count", "particles"))
    count > 0 || error("Profile particle count must be positive")
    seed = Int(get(particles, "seed", 1))
    rng = MersenneTwister(seed)
    total_mass = Float64(get(particles, "total_mass", 1.0e15))
    radius_min = Float64(get(particles, "radius_min", 2.0e4))
    radius_max = Float64(get(particles, "radius_max", 8.0e4))
    z_half_width = Float64(get(particles, "z_half_width", 1.0e3))

    positions = zeros(3, count)
    velocities = zeros(3, count)
    masses = fill(total_mass / count, count)
    for index in 1:count
        angle = 2π * rand(rng)
        radius = radius_min + (radius_max - radius_min) * rand(rng)
        positions[:, index] .= (radius * cos(angle), radius * sin(angle),
                                (2rand(rng) - 1) * z_half_width)
        speed = sqrt(gravitational_constant * total_mass / radius)
        velocities[:, index] .= (-speed * sin(angle), speed * cos(angle), 0.0)
    end
    return positions, velocities, masses
end

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

function load_profile(path::AbstractString)
    data = TOML.parsefile(path)
    simulation = get(data, "simulation", Dict{String, Any}())
    particles = get(data, "particles", Dict{String, Any}())
    gravitational_constant = Float64(get(simulation, "gravitational_constant", 6.67430e-11))
    smoothing = Float64(get(simulation, "smoothing", 100.0))
    particle_radius = Float64(get(simulation, "particle_radius", 0.0))
    opening_angle = Float64(get(simulation, "opening_angle", 0.5))
    timestep = Float64(get(simulation, "timestep", 1.0))
    num_steps = Int(get(simulation, "steps", 120))
    verification_enabled = Bool(get(simulation, "verification_enabled", true))
    verification_energy_max_particles = Int(get(simulation, "verification_energy_max_particles", 2000))
    video_encoding_enabled = Bool(get(simulation, "video_encoding_enabled", true))
    fps = Float64(get(simulation, "fps", 1.0))

    gravitational_constant > 0 || error("Profile gravitational_constant must be positive")
    smoothing >= 0 || error("Profile smoothing must be non-negative")
    particle_radius >= 0 || error("Profile particle_radius must be non-negative")
    opening_angle > 0 || error("Profile opening_angle must be positive")
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
        positions, velocities, masses = _generated_particles(particles, gravitational_constant)
    end

    all(masses .> 0) || error("Profile masses must be positive")
    return SimulationProfile(gravitational_constant, smoothing, particle_radius, opening_angle,
                             timestep, num_steps, verification_enabled,
                             verification_energy_max_particles, video_encoding_enabled, fps,
                             positions, velocities, masses)
end