using DrWatson
@quickactivate "AHRB"

using AbstractTrees, AdaptiveHierarchicalRegularBinning, BenchmarkTools, Random, DataFrames, ColorSchemes, Colors, ProgressMeter, CairoMakie
using Printf
using MPI

const G = 6.67430e-11

function rand_particles(num_particles::Int64)
    pos  = zeros(3, num_particles)
    vel  = zeros(3, num_particles)
    mass = zeros(num_particles)

    # Realistic kilometer-scale compact system: ~10^15 kg spread across ~10^5 m.
    total_mass = 1.0e15
    p_mass = total_mass / num_particles

    for i = 1:num_particles
        θ = 2π * rand()
        R = 2.0e4 + 6.0e4 * rand()
        z = (rand() - 0.5) * 2.0e3

        # Keplerian orbital velocity v = √(G * M / R)
        v = √(G * total_mass / R)

        pos[:, i] .= [R * cos(θ), R * sin(θ), z]
        # Counter-clockwise velocity
        vel[:, i] .= [-v * sin(θ), v * cos(θ), 0.0]
        mass[i] = p_mass
    end
    return (pos, vel, mass)
end

getmass(node::SpatialTree) = getcontext(node)[:mass]
getcom(node::SpatialTree)  = getcontext(node)[:com]

function grav_acc(mass::Float64, r::Vector{Float64}; ϵ::Float64 = 100.0)
    d2 = sum(r.^2) + ϵ^2
    return ((G * mass) / (d2^(1.5))) .* r
end

function net_acc(pos::Vector{Float64}, vel::Vector{Float64}, mass::Float64, node::SpatialTree, tree::SpatialTree, θ::Float64, all_masses::Vector{Float64})
    s = sidelength(node)
    r = getcom(node) .- pos

    if isleaf(node)
        pp  = points(node)
        idx = range(node)
        acc = zeros(3)

        for i in axes(pp, 2)
            idx_orig = tree.info.perm[idx[i]]
            r_vec = pp[:, i] .- pos

            # Skip self-interaction
            if sum(r_vec.^2) > 1e-12
                acc .+= grav_acc(all_masses[idx_orig], r_vec)
            end
        end
        return acc
    elseif s / √(sum(r.^2)) < θ
        return grav_acc(getmass(node), r)
    else
        return sum(net_acc(pos, vel, mass, child, tree, θ, all_masses) for child in children(node))
    end
end

function simulation_step!(pos::Matrix{Float64}, vel::Matrix{Float64}, masses::Vector{Float64}, tree::SpatialTree, Δt::Float64, θ::Float64)
    n_particles = size(pos, 2)
    accs = zeros(3, n_particles)
    
    # Calculate all forces first (Thread-safe)
    Threads.@threads for i in 1:n_particles
        accs[:, i] = net_acc(pos[:, i], vel[:, i], masses[i], tree, tree, θ, masses)
    end
    
    # Update positions and velocities after all forces are known
    for i in 1:n_particles
        vel[:, i] .+= accs[:, i] .* Δt
        pos[:, i] .+= vel[:, i] .* Δt
    end
end

function mpi_local_range(n_particles::Int, rank::Int, n_ranks::Int)
    base, remainder = divrem(n_particles, n_ranks)
    first_particle = rank * base + min(rank, remainder) + 1
    last_particle = first_particle + base - 1 + (rank < remainder)
    return first_particle:last_particle
end

function simulation_step_mpi!(pos::Matrix{Float64}, vel::Matrix{Float64}, masses::Vector{Float64},
                             tree::SpatialTree, Δt::Float64, θ::Float64, comm, rank::Int, n_ranks::Int)
    local_pos = zeros(size(pos))
    local_vel = zeros(size(vel))
    local_range = mpi_local_range(size(pos, 2), rank, n_ranks)

    Threads.@threads for i in local_range
        acc = net_acc(pos[:, i], vel[:, i], masses[i], tree, tree, θ, masses)
        local_vel[:, i] .= vel[:, i] .+ acc .* Δt
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
# TREE INITIALIZATION HELPER
# ---------------------------------------------------------
function update_mass_com!(tree::SpatialTree, masses::Vector{Float64})
    foreach(PostOrderDFS(tree)) do node
        if isleaf(node)
            p_idx = tree.info.perm[range(node)]
            m_sum = sum(@view masses[p_idx])
            pts   = points(node)
            c_m   = m_sum > 0 ? (pts * masses[p_idx]) ./ m_sum : zeros(3)
            setcontext!(node, (; com = c_m, mass = m_sum))
        else
            c_m   = zeros(3)
            m_sum = 0.0
            for child in children(node)
                c_m   .+= getcontext(child)[:com] .* getcontext(child)[:mass]
                m_sum  += getcontext(child)[:mass]
            end
            c_m ./= (m_sum > 0 ? m_sum : 1.0)
            setcontext!(node, (; com = c_m, mass = m_sum))
        end
    end
end

# ---------------------------------------------------------
# MAIN SIMULATION
# ---------------------------------------------------------
frames = Vector{Matrix{Float64}}()
num_particles = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 25000
num_steps = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 120

pos, vel, masses = MPI_RANK == 0 ? rand_particles(num_particles) :
                                  (zeros(3, num_particles), zeros(3, num_particles), zeros(num_particles))
MPI.Bcast!(pos, 0, MPI_COMM)
MPI.Bcast!(vel, 0, MPI_COMM)
MPI.Bcast!(masses, 0, MPI_COMM)

if MPI_RANK == 0
    push!(frames, copy(pos)) # Push initial state (t=0)
end

if MPI_RANK == 0
    println("MPI ranks=$(MPI_SIZE) | threads=$(Threads.nthreads()) | particles=$num_particles | steps=$num_steps")
    flush(stdout)
end

log_interval = max(1, num_steps ÷ 10) # Log every 10%

simulation_time = @elapsed begin
    for step in 1:num_steps
        tree = ahrb(pos, 10, 4; ctxtype = NamedTuple{(:com, :mass), Tuple{Vector{Float64}, Float64}})

        update_mass_com!(tree, masses)
        simulation_step_mpi!(pos, vel, masses, tree, 1.0, 0.5, MPI_COMM, MPI_RANK, MPI_SIZE)
        if MPI_RANK == 0
            push!(frames, copy(pos))
        end
        
        # Print progress and force write to SLURM logs
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
# Ensure the output directory exists
out_dir = "output"
mkpath(out_dir)

# Define the full path using string interpolation
video_filename = "bh-animation_$(num_particles)p_$(num_steps)s.mp4"
out_file = joinpath(out_dir, video_filename)

println("Setting up CairoMakie animation...")
flush(stdout)

fig = Figure(size = (1000, 800))

# Calculate fixed axis limits based on the final cloud size
max_r = max(maximum(abs, frames[end]) * 1.1, 0.5)

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
             perspectiveness = 0.5)
       for col in 1:2]

for ax in axs
    hidedecorations!(ax)
    hidespines!(ax)
end

# Create Observables (Reactive Variables)
x_obs = Observable(frames[1][1, :])
y_obs = Observable(frames[1][2, :])
z_obs = Observable(frames[1][3, :])

# Draw the initial scatter plot into both axes
for ax in axs
    scatter!(ax, x_obs, y_obs, z_obs, color = (:black, 0.4), markersize = 3)
end

total_frames = length(frames)
vid_log_interval = max(1, total_frames ÷ 10)

println("Starting video encoding to $out_file ...")
flush(stdout)

video_encoding_time = @elapsed begin
    record(fig, out_file, 1:total_frames; framerate = 1) do i
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
        println(io, "type,node_count,core_count,particle_count,steps,simulation_time,encoding_time")
    end
    simulation_time_string = @sprintf("%.2f", simulation_time)
    encoding_time_string = @sprintf("%.2f", video_encoding_time)
    println(io, "B_H,$node_count,$core_count,$num_particles,$num_steps,$simulation_time_string,$encoding_time_string")
end
end

MPI.Barrier(MPI_COMM)
MPI.Finalize()
