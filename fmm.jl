using DrWatson
@quickactivate "AHRB"

using BenchmarkTools, Random, DataFrames, ColorSchemes, Colors, ProgressMeter, CairoMakie
using Printf
using MPI
using FastMultipole
using FastMultipole.StaticArrays: SVector, SMatrix

"""
    rand_particles(num_particles::Int64)

Generate `num_particles` random particles using a distribution similar to a disk-shaped galaxy.
"""
function rand_particles(num_particles::Int64)
    pos  = zeros(3, num_particles)
    vel  = zeros(3, num_particles)
    mass = zeros(num_particles)

    total_mass = 1.0
    p_mass = total_mass / num_particles

    for i = 1:num_particles
        θ = 2π * rand()
        R = 0.1 + 0.4 * rand()
        z = (rand() - 0.5) * 0.02

        # Keplerian orbital velocity v = √(G * M / R)
        v = √(total_mass / R)

        pos[:, i] .= [R * cos(θ), R * sin(θ), z]
        # Counter-clockwise velocity
        vel[:, i] .= [-v * sin(θ), v * cos(θ), 0.0]
        mass[i] = p_mass
    end
    return (pos, vel, mass)
end

struct FMMBody{T}
    position::SVector{3,T}
    radius::T
    strength::T
end

mutable struct GravitationalSystem{T}
    bodies::Vector{FMMBody{T}}
    potential::Matrix{T}
end

function GravitationalSystem(pos::Matrix{T}, masses::Vector{T}; radius::T = 0.02) where {T <: AbstractFloat}
    bodies = [FMMBody(SVector{3,T}(pos[:, i]), radius, masses[i]) for i in axes(pos, 2)]
    return GravitationalSystem(bodies, zeros(T, 16, length(bodies)))
end

Base.eltype(::GravitationalSystem{T}) where {T} = T

function FastMultipole.source_system_to_buffer!(buffer, i_buffer, system::GravitationalSystem, i_body)
    x, y, z = system.bodies[i_body].position
    buffer[1, i_buffer] = x
    buffer[2, i_buffer] = y
    buffer[3, i_buffer] = z
    buffer[4, i_buffer] = system.bodies[i_body].radius
    buffer[5, i_buffer] = system.bodies[i_body].strength
end

FastMultipole.data_per_body(::GravitationalSystem) = 5
FastMultipole.get_position(system::GravitationalSystem, i) = system.bodies[i].position
FastMultipole.strength_dims(::GravitationalSystem) = 1
FastMultipole.get_n_bodies(system::GravitationalSystem) = length(system.bodies)
FastMultipole.has_vector_potential(::GravitationalSystem) = false

# API FIX 1: Fully qualified Point/Source, removed unsupported `scale_strength` keyword
FastMultipole.body_to_multipole!(system::GravitationalSystem, args...) =
    FastMultipole.body_to_multipole!(FastMultipole.Point{FastMultipole.Source}, system, args...)

# API FIX 2: Target buffer parameter corrected, `switch` argument removed from set_gradient!
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
                r = sqrt(r2)
                gradient -= SVector{3}(dx, dy, dz) * source_strength / (4π * r2 * r)
            end
        end
        GS && FastMultipole.set_gradient!(target_buffer, j_target, gradient)
    end
end

# API FIX 3: `switch` argument removed from get_gradient
function FastMultipole.buffer_to_target_system!(target_system::GravitationalSystem, i_target,
                                                switch::FastMultipole.DerivativesSwitch{PS,GS,HS},
                                                target_buffer, i_buffer) where {PS,GS,HS}
    gradient = GS ? FastMultipole.get_gradient(target_buffer, i_buffer) : zero(SVector{3,eltype(target_system)})
    target_system.potential[5:7, i_target] .= gradient
end

function simulation_step!(pos::Matrix{Float64}, vel::Matrix{Float64}, masses::Vector{Float64}, Δt::Float64)
    system = GravitationalSystem(pos, masses)
    
    # API FIX 4: Silenced the missing 'get_previous_influence' warning natively
    fmm!(system; gradient = true, silence_warnings = true)
    
    accs = @view system.potential[5:7, :]
    
    # PHYSICS FIX: Because we had to drop `scale_strength = -1.0` in FIX 1, we subtract 
    # the acceleration to ensure gravity remains attractive rather than repulsive.
    vel .-= accs .* Δt
    pos .+= vel .* Δt
end

function mpi_local_range(n_particles::Int, rank::Int, n_ranks::Int)
    base, remainder = divrem(n_particles, n_ranks)
    first_particle = rank * base + min(rank, remainder) + 1
    last_particle = first_particle + base - 1 + (rank < remainder)
    return first_particle:last_particle
end

function simulation_step_mpi!(pos::Matrix{Float64}, vel::Matrix{Float64}, masses::Vector{Float64},
                             Δt::Float64, comm, rank::Int, n_ranks::Int)
    system = GravitationalSystem(pos, masses)
    fmm!(system; gradient = true, silence_warnings = true)

    local_pos = zeros(size(pos))
    local_vel = zeros(size(vel))
    local_range = mpi_local_range(size(pos, 2), rank, n_ranks)
    accs = @view system.potential[5:7, :]

    Threads.@threads for i in local_range
        local_vel[:, i] .= vel[:, i] .- accs[:, i] .* Δt
        local_pos[:, i] .= pos[:, i] .+ local_vel[:, i] .* Δt
    end

    MPI.Allreduce!(local_pos, pos, +, comm)
    MPI.Allreduce!(local_vel, vel, +, comm)
end

MPI.Init()
const MPI_COMM = MPI.COMM_WORLD
const MPI_RANK = MPI.Comm_rank(MPI_COMM)
const MPI_SIZE = MPI.Comm_size(MPI_COMM)

# ---------------------------------------------------------
# WARM-UP COMPILATION SPINNER
# ---------------------------------------------------------
function compile_with_spinner(func::Function, message::String)
    done = Threads.Atomic{Bool}(false)
    spin_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    
    spinner_task = Threads.@spawn begin
        i = 1
        while !done[]
            print("\r\033[K$message ", spin_chars[i])
            sleep(0.1)
            i = (i % length(spin_chars)) + 1
        end
        print("\r\033[K$message Done! ✨\n")
    end
    
    func() 
    done[] = true 
    wait(spinner_task)
end

warmup = () -> begin
    warm_pos, warm_vel, warm_mass = rand(3, 2), rand(3, 2), [0.5, 0.5]
    simulation_step!(warm_pos, warm_vel, warm_mass, 0.0005)
end

if MPI_RANK == 0
    compile_with_spinner(warmup, "Compiling physics functions...")
else
    warmup()
end

# ---------------------------------------------------------
# MAIN SIMULATION
# ---------------------------------------------------------
frames = Vector{Matrix{Float64}}()
num_particles = 1000
num_steps = 1000

pos, vel, masses = MPI_RANK == 0 ? rand_particles(num_particles) :
                                  (zeros(3, num_particles), zeros(3, num_particles), zeros(num_particles))
MPI.Bcast!(pos, 0, MPI_COMM)
MPI.Bcast!(vel, 0, MPI_COMM)
MPI.Bcast!(masses, 0, MPI_COMM)

if MPI_RANK == 0
    push!(frames, copy(pos)) # Push initial state (t=0)
end

if MPI_RANK == 0
    println("Starting Galaxy Simulation with $MPI_SIZE MPI ranks...")
    flush(stdout)
end

log_interval = max(1, num_steps ÷ 10) # Log every 10%

simulation_time = @elapsed begin
    for step in 1:num_steps
        simulation_step_mpi!(pos, vel, masses, 0.0005, MPI_COMM, MPI_RANK, MPI_SIZE)
        if MPI_RANK == 0
            push!(frames, copy(pos))
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
    flush(stdout)
end

if MPI_RANK == 0
# ---------------------------------------------------------
# CAIROMAKIE VIDEO EXPORT
# ---------------------------------------------------------
out_dir = "output"
mkpath(out_dir)

video_filename = "bh-animation_$(num_particles)p_$(num_steps)s.mp4"
out_file = joinpath(out_dir, video_filename)

println("Setting up CairoMakie animation...")
flush(stdout)

fig = Figure(size = (1000, 800))

# Calculate fixed axis limits based on the final cloud size
max_r = max(maximum(abs, frames[end]) * 1.1, 0.5)

# Viewing angles (azimuth, elevation) converted to radians for Makie
angles = [
    (deg2rad(30.0), deg2rad(30.0))  (deg2rad(10.0), deg2rad(80.0));
    (deg2rad(80.0), deg2rad(10.0))  (deg2rad(60.0), deg2rad(30.0))
]

# Create the 2x2 grid of 3D axes
axs = [Axis3(fig[row, col], 
             azimuth = angles[row, col][1], 
             elevation = angles[row, col][2],
             limits = (-max_r, max_r, -max_r, max_r, -max_r, max_r),
             aspect = :data,
             perspectiveness = 0.5)
       for row in 1:2, col in 1:2]

for ax in axs
    hidedecorations!(ax)
    hidespines!(ax)
end

# Create Observables (Reactive Variables)
x_obs = Observable(frames[1][1, :])
y_obs = Observable(frames[1][2, :])
z_obs = Observable(frames[1][3, :])

# Draw the initial scatter plot into all 4 axes
for ax in axs
    scatter!(ax, x_obs, y_obs, z_obs, color = (:black, 0.4), markersize = 3)
end

total_frames = length(frames)
vid_log_interval = max(1, total_frames ÷ 10)

println("Starting video encoding to $out_file ...")
flush(stdout)

video_encoding_time = @elapsed begin
    record(fig, out_file, 1:total_frames; framerate = 20) do i
        x_obs[] = frames[i][1, :]
        y_obs[] = frames[i][2, :]
        z_obs[] = frames[i][3, :]

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
        println(io, "type,node_count,core_count,simulation_time,encoding_time")
    end
    simulation_time_string = @sprintf("%.2f", simulation_time)
    encoding_time_string = @sprintf("%.2f", video_encoding_time)
    println(io, "FMM,$node_count,$core_count,$simulation_time_string,$encoding_time_string")
end
end

MPI.Barrier(MPI_COMM)
MPI.Finalize()
