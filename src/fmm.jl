using DrWatson
using BenchmarkTools, Random, DataFrames, ColorSchemes, Colors, ProgressMeter, CairoMakie
using Printf
using MPI
using FastMultipole
using FastMultipole.StaticArrays: SVector, SMatrix

if !@isdefined __NBodySimCommonLoaded
    include(joinpath(@__DIR__, "simulation_common.jl"))
    global __NBodySimCommonLoaded = true
end

# FastMultipole body wrapper.
struct FMMBody{T}
    position::SVector{3,T}
    radius::T
    strength::T
end

# FMM source/target container.
mutable struct GravitationalSystem{T}
    bodies::Vector{FMMBody{T}}
    potential::Matrix{T}
    interaction_strength::T
    smoothing::T
end

function GravitationalSystem(pos::Matrix{T}, masses::Vector{T}; particle_radius::T,
                             smoothing::T, interaction_strength::T) where {T <: AbstractFloat}
    bodies = [FMMBody(SVector{3,T}(pos[:, i]), particle_radius, masses[i]) for i in axes(pos, 2)]
    return GravitationalSystem(bodies, zeros(T, 16, length(bodies)), interaction_strength, smoothing)
end

Base.eltype(::GravitationalSystem{T}) where {T} = T

# Pack each source body into the FMM buffer.
function FastMultipole.source_system_to_buffer!(buffer, i_buffer, system::GravitationalSystem, i_body)
    x, y, z = system.bodies[i_body].position
    buffer[1, i_buffer] = x
    buffer[2, i_buffer] = y
    buffer[3, i_buffer] = z
    buffer[4, i_buffer] = system.bodies[i_body].radius
    # FastMultipole's Laplace kernel is normalized by 1 / (4*pi). Scale the
    # source strength so its gradient has physical gravitational units.
    buffer[5, i_buffer] = 4π * system.interaction_strength * system.bodies[i_body].strength
end

FastMultipole.data_per_body(::GravitationalSystem) = 5
FastMultipole.get_position(system::GravitationalSystem, i) = system.bodies[i].position
FastMultipole.strength_dims(::GravitationalSystem) = 1
FastMultipole.get_n_bodies(system::GravitationalSystem) = length(system.bodies)
FastMultipole.has_vector_potential(::GravitationalSystem) = false

FastMultipole.body_to_multipole!(system::GravitationalSystem, args...) =
    FastMultipole.body_to_multipole!(FastMultipole.Point{FastMultipole.Source}, system, args...)

# Direct FMM interaction for each target/source pair.
function FastMultipole.direct!(target_buffer, target_index,
                              switch::FastMultipole.DerivativesSwitch{PS,GS,HS},
                              source_system::GravitationalSystem, source_buffer,
                              source_index) where {PS,GS,HS}
    @inbounds for j_target in target_index
        target_x, target_y, target_z = FastMultipole.get_position(target_buffer, j_target)
        gradient = zero(SVector{3,eltype(target_buffer)})
        @inbounds for i_source in source_index
            source_x, source_y, source_z = FastMultipole.get_position(source_buffer, i_source)
            source_strength = FastMultipole.get_strength(source_buffer, source_system, i_source)[1]
            dx, dy, dz = target_x - source_x, target_y - source_y, target_z - source_z
            r2 = dx * dx + dy * dy + dz * dz
            if r2 > 0
                softened_r2 = r2 + source_system.smoothing^2
                gradient -= SVector{3}(dx, dy, dz) * source_strength /
                            (4π * softened_r2 * sqrt(softened_r2))
            end
        end
        GS && FastMultipole.set_gradient!(target_buffer, j_target, gradient)
    end
end

function FastMultipole.buffer_to_target_system!(target_system::GravitationalSystem, i_target,
                                                switch::FastMultipole.DerivativesSwitch{PS,GS,HS},
                                                target_buffer, i_buffer) where {PS,GS,HS}
    gradient = GS ? FastMultipole.get_gradient(target_buffer, i_buffer) : zero(SVector{3,eltype(target_system)})
    target_system.potential[5:7, i_target] .= gradient
end

# Compute local FMM accelerations and reduce the global state.
function simulation_step_fmm_mpi!(pos::Matrix{Float64}, vel::Matrix{Float64}, masses::Vector{Float64},
                             profile::SimulationProfile, comm, rank::Int, n_ranks::Int)
    local_pos = zeros(size(pos))
    local_vel = zeros(size(vel))
    local_range = mpi_local_range(size(pos, 2), rank, n_ranks)
    source_system = GravitationalSystem(pos, masses; particle_radius = profile.particle_radius,
                                        smoothing = profile.smoothing,
                                        interaction_strength = profile.interaction_strength)
    target_system = GravitationalSystem(pos[:, local_range], masses[local_range];
                                        particle_radius = profile.particle_radius,
                                        smoothing = profile.smoothing,
                                        interaction_strength = profile.interaction_strength)
    fmm!(target_system, source_system; gradient = true,
            expansion_order = profile.fmm.expansion_order,
            multipole_acceptance = profile.fmm.multipole_acceptance,
            silence_warnings = true)

    accs = @view target_system.potential[5:7, :]

    Threads.@threads for local_index in eachindex(local_range)
        global_index = local_range[local_index]
        local_vel[:, global_index] .= vel[:, global_index] .+ accs[:, local_index] .* profile.timestep
        local_pos[:, global_index] .= pos[:, global_index] .+ local_vel[:, global_index] .* profile.timestep
    end

    MPI.Allreduce!(local_pos, pos, +, comm)
    MPI.Allreduce!(local_vel, vel, +, comm)
end

function run_fmm_simulation(profile_path::AbstractString="profiles/default.toml",
                           num_steps_override::Union{Nothing,Int}=nothing)
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    n_ranks = MPI.Comm_size(comm)

    if haskey(ENV, "SLURM_NTASKS")
        expected_mpi_size = parse(Int, ENV["SLURM_NTASKS"])
        n_ranks == expected_mpi_size || error(
            "MPI world has $n_ranks ranks, but Slurm allocated $expected_mpi_size tasks. " *
            "Launch this job with an MPI-enabled srun command."
        )
    end

    frames = Vector{Matrix{Float64}}()
    vel_frames = Vector{Matrix{Float64}}()
    profile_file = isabspath(profile_path) ? profile_path : joinpath(dirname(@__DIR__), profile_path)
    profile = load_profile(profile_file)
    num_particles = size(profile.positions, 2)
    num_steps = num_steps_override === nothing ? profile.num_steps : Int(num_steps_override)
    verification_baseline = profile.verification_enabled ?
        verification_reference(profile.positions, profile.velocities, profile.masses,
                               profile.interaction_strength, profile.smoothing,
                               profile.verification_energy_max_particles) : nothing
    verification_rows = rank == 0 && profile.verification_enabled ?
        [(0, 0.0, verification_metrics(profile.positions, profile.velocities,
                                        profile.masses, profile.interaction_strength,
                                        profile.smoothing, verification_baseline,
                                        profile.verification_energy_max_particles))] : nothing

    pos, vel, masses = rank == 0 ? (copy(profile.positions), copy(profile.velocities), copy(profile.masses)) :
                                    (zeros(3, num_particles), zeros(3, num_particles), zeros(num_particles))
    MPI.Bcast!(pos, 0, comm)
    MPI.Bcast!(vel, 0, comm)
    MPI.Bcast!(masses, 0, comm)

    if rank == 0
        if profile.video_encoding_enabled || profile.store_data
            push!(frames, copy(pos))
        end
        if profile.store_data
            push!(vel_frames, copy(vel))
        end
    end

    if rank == 0
        println("MPI ranks=$(n_ranks) | threads=$(Threads.nthreads()) | particles=$num_particles | steps=$num_steps")
        flush(stdout)
    end

    log_interval = max(1, num_steps ÷ 10)

    simulation_time = @elapsed begin
        for step in 1:num_steps
            simulation_step_fmm_mpi!(pos, vel, masses, profile, comm, rank, n_ranks)
            if rank == 0
                if profile.video_encoding_enabled || profile.store_data
                    push!(frames, copy(pos))
                end
                if profile.store_data
                    push!(vel_frames, copy(vel))
                end
            end
            if rank == 0 && profile.verification_enabled
                if step % log_interval == 0 || step == num_steps
                    push!(verification_rows, (step, step * profile.timestep,
                                              verification_metrics(pos, vel, masses,
                                                                   profile.interaction_strength,
                                                                   profile.smoothing, verification_baseline,
                                                                   profile.verification_energy_max_particles)))
                end
            end

            if rank == 0 && step % log_interval == 0
                percent = round(Int, (step / num_steps) * 100)
                println("Simulation progress: $step / $num_steps steps ($percent%)")
                flush(stdout)
            end
        end
    end

    if rank == 0
        println("Total simulation time: $simulation_time seconds")
        if profile.verification_enabled
            verification_file = joinpath("logs", "verification_fmm_$(num_particles)p_$(num_steps)s.csv")
            mkpath(dirname(verification_file))
            open(verification_file, "w") do io
                write_verification_header(io)
                for (step, time, metrics) in verification_rows
                    write_verification_row(io, step, time, metrics)
                end
            end
            final_metrics = verification_rows[end][3]
            println("Verification: COM drift=$(final_metrics.center_of_mass_drift), " *
                    "momentum drift=$(final_metrics.relative_momentum_change), " *
                    "angular momentum drift=$(final_metrics.relative_angular_momentum_change), " *
                    "relative energy change=$(final_metrics.relative_energy_change)")
            println("Verification metrics saved to $verification_file")
        end
        flush(stdout)
    end

    if rank == 0 && profile.video_encoding_enabled
        encode_video_and_log(frames, masses, num_particles, num_steps, round(Int, profile.fps), simulation_time, "FMM", "fmm")
    end

    if rank == 0 && profile.store_data
        save_simulation_hdf5(frames, vel_frames, masses, profile, num_particles, num_steps, "FMM")
    end

    MPI.Barrier(comm)
    MPI.Finalize()
    return nothing
end

function main_fmm(args::Vector{String}=ARGS)
    profile_path = length(args) >= 1 ? args[1] : "profiles/default.toml"
    num_steps = length(args) >= 2 ? parse(Int, args[2]) : nothing
    run_fmm_simulation(profile_path, num_steps)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_fmm(ARGS)
end