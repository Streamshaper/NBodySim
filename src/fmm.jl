using DrWatson
using BenchmarkTools, Random, DataFrames, ColorSchemes, Colors, ProgressMeter, CairoMakie
using Printf
using MPI
using FastMultipole
using FastMultipole.StaticArrays: SVector, SMatrix
include(joinpath(@__DIR__, "profile.jl"))
include(joinpath(@__DIR__, "verification.jl"))

struct FMMBody{T}
    position::SVector{3,T}
    radius::T
    strength::T
end

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

function simulation_step!(pos::Matrix{Float64}, vel::Matrix{Float64}, masses::Vector{Float64}, profile::SimulationProfile)
    system = GravitationalSystem(pos, masses; particle_radius = profile.particle_radius,
                                 smoothing = profile.smoothing,
                                 interaction_strength = profile.interaction_strength)
    
    fmm!(system; gradient = true,
        expansion_order = profile.fmm.expansion_order,
        multipole_acceptance = profile.fmm.multipole_acceptance,
        leaf_size = profile.fmm.leaf_size,
        silence_warnings = true) 

    accs = @view system.potential[5:7, :]
    
    # FastMultipole returns the inward gravitational gradient for positive masses.
    vel .+= accs .* profile.timestep
    pos .+= vel .* profile.timestep
end

function mpi_local_range(n_particles::Int, rank::Int, n_ranks::Int)
    base, remainder = divrem(n_particles, n_ranks)
    first_particle = rank * base + min(rank, remainder) + 1
    last_particle = first_particle + base - 1 + (rank < remainder)
    return first_particle:last_particle
end

function simulation_step_mpi!(pos::Matrix{Float64}, vel::Matrix{Float64}, masses::Vector{Float64},
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

MPI.Init()
const MPI_COMM = MPI.COMM_WORLD
const MPI_RANK = MPI.Comm_rank(MPI_COMM)
const MPI_SIZE = MPI.Comm_size(MPI_COMM)

if haskey(ENV, "SLURM_NTASKS")
    expected_mpi_size = parse(Int, ENV["SLURM_NTASKS"])
    MPI_SIZE == expected_mpi_size || error(
        "MPI world has $MPI_SIZE ranks, but Slurm allocated $expected_mpi_size tasks. " *
        "Launch this job with an MPI-enabled srun command."
    )
end

# ---------------------------------------------------------
# MAIN SIMULATION
# ---------------------------------------------------------
frames = Vector{Matrix{Float64}}()
profile_path = length(ARGS) >= 1 ? ARGS[1] : "profiles/default.toml"
profile_file = isabspath(profile_path) ? profile_path : joinpath(dirname(@__DIR__), profile_path)
profile = load_profile(profile_file)
num_particles = size(profile.positions, 2)
num_steps = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : profile.num_steps
verification_baseline = profile.verification_enabled ?
    verification_reference(profile.positions, profile.velocities, profile.masses,
                           profile.interaction_strength, profile.smoothing,
                           profile.verification_energy_max_particles) : nothing
verification_rows = MPI_RANK == 0 && profile.verification_enabled ?
    [(0, 0.0, verification_metrics(profile.positions, profile.velocities,
                                    profile.masses, profile.interaction_strength,
                                    profile.smoothing, verification_baseline,
                                    profile.verification_energy_max_particles))] : nothing

pos, vel, masses = MPI_RANK == 0 ? (copy(profile.positions), copy(profile.velocities), copy(profile.masses)) :
                                  (zeros(3, num_particles), zeros(3, num_particles), zeros(num_particles))
MPI.Bcast!(pos, 0, MPI_COMM)
MPI.Bcast!(vel, 0, MPI_COMM)
MPI.Bcast!(masses, 0, MPI_COMM)


if MPI_RANK == 0 && profile.video_encoding_enabled
    push!(frames, copy(pos)) # Push initial state (t=0)
end

if MPI_RANK == 0
    println("MPI ranks=$(MPI_SIZE) | threads=$(Threads.nthreads()) | particles=$num_particles | steps=$num_steps")
    flush(stdout)
end

log_interval = max(1, num_steps ÷ 10) # Log every 10%

simulation_time = @elapsed begin
    for step in 1:num_steps
        simulation_step_mpi!(pos, vel, masses, profile, MPI_COMM, MPI_RANK, MPI_SIZE)
        if MPI_RANK == 0 && profile.video_encoding_enabled
            push!(frames, copy(pos))
        end
        if MPI_RANK == 0 && profile.verification_enabled
            if step % log_interval == 0 || step == num_steps
                push!(verification_rows, (step, step * profile.timestep,
                                          verification_metrics(pos, vel, masses,
                                                               profile.interaction_strength,
                                                               profile.smoothing, verification_baseline,
                                                               profile.verification_energy_max_particles)))
            end
        end

        # Print progress and force write to logs
        if MPI_RANK == 0 && step % log_interval == 0
            percent = round(Int, (step / num_steps) * 100)
            println("Simulation progress: $step / $num_steps steps ($percent%)")
            flush(stdout)
        end
    end
end

if MPI_RANK == 0
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

if MPI_RANK == 0 && profile.video_encoding_enabled

# ---------------------------------------------------------
# CAIROMAKIE VIDEO EXPORT
# ---------------------------------------------------------
out_dir = "output"
mkpath(out_dir)

video_filename = "fmm-animation_$(num_particles)p_$(num_steps)s.mp4"
out_file = joinpath(out_dir, video_filename)

println("Setting up CairoMakie animation...")
flush(stdout)

fig = Figure(size = (1400, 1000), backgroundcolor = :black)

all(frame -> all(isfinite, frame), frames) ||
    error("Cannot encode video: simulation produced non-finite particle positions")
plot_scale = maximum(maximum(abs, frame) for frame in frames)
isfinite(plot_scale) && plot_scale > 0 || error("Cannot encode video: particle positions exceed finite plotting limits")
max_r = 1.05

# Viewing angles (azimuth, elevation) converted to radians for Makie
angles = [
    (deg2rad(0.0), deg2rad(90.0)),
    (deg2rad(60.0), deg2rad(30.0))
]

# Create the 1x2 side-by-side grid of 3D axes
axs = [Axis3(fig[1, col], 
             azimuth = angles[col][1], 
             elevation = angles[col][2],
             limits = (-max_r, max_r, -max_r, max_r, -max_r, max_r),
             aspect = :data,
             perspectiveness = 0.5,
             backgroundcolor = :black)
       for col in 1:2]

for ax in axs
    hidedecorations!(ax)
    hidespines!(ax)
end

# Create Observables (Reactive Variables)
x_obs = Observable(frames[1][1, :] ./ plot_scale)
y_obs = Observable(frames[1][2, :] ./ plot_scale)
z_obs = Observable(frames[1][3, :] ./ plot_scale)
mass_scale = cbrt.(masses ./ maximum(masses))
marker_sizes = 1.5 .+ 9.0 .* mass_scale

# Draw the initial scatter plot into both axes
for ax in axs
    scatter!(ax, x_obs, y_obs, z_obs, color = (:white, 0.5), markersize = marker_sizes)
end

total_frames = length(frames)
vid_log_interval = max(1, total_frames ÷ 10)

println("Starting video encoding to $out_file ...")
flush(stdout)

video_encoding_time = @elapsed begin
    record(fig, out_file, 1:total_frames; framerate = profile.fps) do i
        x_obs[] = frames[i][1, :] ./ plot_scale
        y_obs[] = frames[i][2, :] ./ plot_scale
        z_obs[] = frames[i][3, :] ./ plot_scale

        # Print encoding progress and force write to SLURM logs
        if i % vid_log_interval == 0
            percent = round(Int, (i / total_frames) * 100)
            println("Encoding video: frame $i / $total_frames ($percent%)")
            flush(stdout)
        end
    end
end

println("Total video encoding time: $video_encoding_time seconds")
println("Saved $out_file successfully!")
flush(stdout)

stopwatch_file = joinpath("logs", "stopwatch.csv")
mkpath(dirname(stopwatch_file))
core_count = parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", string(Sys.CPU_THREADS)))
node_count = parse(Int, get(ENV, "SLURM_JOB_NUM_NODES", "1"))

open(stopwatch_file, "a+") do io
    if filesize(stopwatch_file) == 0
        println(io, "type,node_count,core_count,particle_count,steps,simulation_time,encoding_time")
    end
    simulation_time_string = @sprintf("%.2f", simulation_time)
    encoding_time_string = @sprintf("%.2f", video_encoding_time)
    println(io, "FMM,$node_count,$core_count,$num_particles,$num_steps,$simulation_time_string,$encoding_time_string")
end
end

MPI.Barrier(MPI_COMM)
MPI.Finalize()
