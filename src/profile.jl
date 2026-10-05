using TOML
using Random

if !@isdefined __NBodySimKernelsLoaded
    include(joinpath(@__DIR__, "kernels.jl"))
    global __NBodySimKernelsLoaded = true
end

if !@isdefined __NBodySimIntegratorsLoaded
    include(joinpath(@__DIR__, "integrators.jl"))
    global __NBodySimIntegratorsLoaded = true
end

if !@isdefined __NBodySimSolversLoaded
    include(joinpath(@__DIR__, "solvers.jl"))
    global __NBodySimSolversLoaded = true
end

# Particle-state file tag.
const PARTICLE_STATE_MAGIC_V1 = UInt8[0x4e, 0x42, 0x53, 0x31]
const PARTICLE_STATE_MAGIC = UInt8[0x4e, 0x42, 0x53, 0x32]

struct FMMProfile
    expansion_order::Int
    multipole_acceptance::Float64
    leaf_size::Int
end

struct SimulationProfile{K<:AbstractKernel, I<:AbstractIntegrator, S<:AbstractSolver}
    kernel::K
    integrator::I
    solver::S
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
    logging_enabled::Bool
    store_data::Bool
    fps::Float64
    positions::Matrix{Float64}
    velocities::Matrix{Float64}
    masses::Vector{Float64}
    charges::Vector{Float64}
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
function _generated_particles(particles, kernel::AbstractKernel)
    kernel isa Coulomb && error("Coulomb profiles require an explicit particle state")
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
    charges = Float64.(get(particles, "charges", zeros(count)))
    length(charges) == count || error("Generated particle charges must match particles.count")
    all(isfinite, charges) || error("Particle charges must be finite")
    
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
        speed = circular_orbital_speed(kernel, central_mass, radius)
        velocities[:, index] .= (-speed * sin(angle), speed * cos(angle), 0.0)
    end
    
    return positions, velocities, masses, charges
end

# Persist a particle state for later reloads.
function write_particle_state(path::AbstractString, positions::Matrix{Float64},
                              velocities::Matrix{Float64}, masses::Vector{Float64};
                              charges::Vector{Float64}=zeros(length(masses)))
    size(positions, 1) == 3 || error("positions must have shape 3 × N")
    size(velocities) == size(positions) || error("positions and velocities must have the same shape")
    length(masses) == size(positions, 2) || error("masses must contain one value per particle")
    length(charges) == size(positions, 2) || error("charges must contain one value per particle")
    all(masses .> 0) || error("Particle masses must be positive")
    all(isfinite, charges) || error("Particle charges must be finite")

    open(path, "w") do io
        write(io, PARTICLE_STATE_MAGIC)
        write(io, UInt64(size(positions, 2)))
        write(io, positions)
        write(io, velocities)
        write(io, masses)
        write(io, charges)
    end
    return path
end

function _load_particle_state(path::AbstractString)
    open(path, "r") do io
        magic = read(io, length(PARTICLE_STATE_MAGIC))
        magic in (PARTICLE_STATE_MAGIC_V1, PARTICLE_STATE_MAGIC) ||
            error("Invalid particle state file '$path'")
        count = Int(read(io, UInt64))
        count > 0 || error("Particle state file must contain at least one particle")

        positions = Matrix{Float64}(undef, 3, count)
        velocities = Matrix{Float64}(undef, 3, count)
        masses = Vector{Float64}(undef, count)
        charges = zeros(count)
        read!(io, positions)
        read!(io, velocities)
        read!(io, masses)
        has_charge_data = magic == PARTICLE_STATE_MAGIC
        has_charge_data && read!(io, charges)
        return positions, velocities, masses, charges, has_charge_data
    end
end

# Load the profile and resolve the particle source.
function load_profile(path::AbstractString;
                      solver_override::Union{Nothing,AbstractSolver}=nothing)
    data = TOML.parsefile(path)
    simulation = get(data, "simulation", Dict{String, Any}())
    kernel_config = get(data, "kernel", Dict{String, Any}())
    integrator_config = get(data, "integrator", Dict{String, Any}())
    solver_config = get(data, "solver", Dict{String, Any}())
    barnes_hut = get(data, "barnes_hut", Dict{String, Any}())
    fmm = get(data, "fmm", Dict{String, Any}())
    particles = get(data, "particles", Dict{String, Any}())
    legacy_strength = get(simulation, "interaction_strength",
                          get(simulation, "gravitational_constant", 6.67430e-11))
    interaction_strength = Float64(get(kernel_config, "interaction_strength",
                                      get(kernel_config, "strength", legacy_strength)))
    smoothing = Float64(get(kernel_config, "smoothing", get(simulation, "smoothing", 100.0)))
    kernel_type = lowercase(String(get(kernel_config, "type", "plummer_gravity")))
    integrator_type = lowercase(String(get(integrator_config, "type", "semi_implicit_euler")))
    configured_solver = parse_solver(String(get(solver_config, "type", "direct")))
    solver = solver_override === nothing ? configured_solver : solver_override
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
    logging_enabled = Bool(get(simulation, "logging_enabled", false))
    store_data = Bool(get(simulation, "store_data", true))
    fps = Float64(get(simulation, "fps", 1.0))

    interaction_strength > 0 || error("Kernel interaction_strength must be positive")
    smoothing >= 0 || error("Kernel smoothing must be non-negative")
    particle_radius >= 0 || error("Profile particle_radius must be non-negative")
    opening_angle > 0 || error("Profile opening_angle must be positive")
    expansion_order > 0 || error("Profile fmm.expansion_order must be positive")
    0 < multipole_acceptance <= 1 || error("Profile fmm.multipole_acceptance must be in (0, 1]")
    leaf_size > 0 || error("Profile fmm.leaf_size must be positive")
    timestep > 0 || error("Profile timestep must be positive")
    num_steps >= 0 || error("Profile steps must be non-negative")
    verification_energy_max_particles >= 0 || error("Profile verification_energy_max_particles must be non-negative")
    fps > 0 || error("Profile fps must be positive")

    kernel = if kernel_type == "plummer_gravity"
        PlummerGravity(interaction_strength, smoothing)
    elseif kernel_type == "yukawa_gravity"
        screening_length = Float64(_required(kernel_config, "screening_length", "kernel"))
        screening_length > 0 || error("Profile kernel.screening_length must be positive")
        YukawaGravity(interaction_strength, smoothing, screening_length)
    elseif kernel_type == "coulomb"
        has_strength = haskey(kernel_config, "interaction_strength") ||
                       haskey(kernel_config, "strength") ||
                       haskey(simulation, "interaction_strength")
        has_strength ||
            error("Coulomb profiles must set an interaction strength in [kernel] or [simulation]")
        Coulomb(interaction_strength, smoothing)
    else
        error("Unknown kernel '$kernel_type'. Expected one of: plummer_gravity, yukawa_gravity, coulomb")
    end
    integrator = if integrator_type == "semi_implicit_euler"
        SemiImplicitEuler()
    else
        error("Unknown integrator '$integrator_type'. Expected one of: semi_implicit_euler")
    end
    require_solver_support(solver, kernel, integrator)

    if haskey(particles, "state_file")
        state_path = String(particles["state_file"])
        state_path = isabspath(state_path) ? state_path : joinpath(dirname(abspath(path)), state_path)
        positions, velocities, masses, charges, has_charge_data = _load_particle_state(state_path)
    elseif haskey(particles, "positions")
        positions = _particle_matrix(particles["positions"], "positions")
        velocities = _particle_matrix(_required(particles, "velocities", "particles"), "velocities")
        masses = Float64.(_required(particles, "masses", "particles"))
        charges = Float64.(get(particles, "charges", zeros(length(masses))))
        has_charge_data = haskey(particles, "charges")
        size(positions) == size(velocities) || error("positions and velocities must have the same shape")
        length(masses) == size(positions, 2) || error("masses must contain one value per particle")
        length(charges) == size(positions, 2) || error("charges must contain one value per particle")
    else
        positions, velocities, masses, charges = _generated_particles(particles, kernel)
        has_charge_data = haskey(particles, "charges")
    end

    all(masses .> 0) || error("Profile masses must be positive")
    all(isfinite, charges) || error("Profile charges must be finite")
    kernel isa Coulomb && !has_charge_data &&
        error("Coulomb profiles require explicit particle charges")
    
    return SimulationProfile(kernel, integrator, solver, interaction_strength, smoothing, particle_radius, opening_angle,
                             FMMProfile(expansion_order, multipole_acceptance, leaf_size),
                             timestep, num_steps, verification_enabled,
                             verification_energy_max_particles, video_encoding_enabled,
                             logging_enabled,
                             store_data, fps,
                             positions, velocities, masses, charges)
end
