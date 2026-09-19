using DrWatson
@quickactivate "AHRB"

using BenchmarkTools, Random, DataFrames, ColorSchemes, Colors, ProgressMeter, CairoMakie
using Printf
using LinearAlgebra
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

const FMM_G = 1.0

mutable struct FMMCell
    center::SVector{3,Float64}
    half_width::Float64
    particle_ids::Vector{Int}
    children::Vector{FMMCell}
    mass::Float64
    center_of_mass::SVector{3,Float64}
    dipole::SVector{3,Float64}
    quadrupole::SMatrix{3,3,Float64}
    local_expansion::SVector{3,Float64}
    interaction_list::Vector{FMMCell}
    is_leaf::Bool
end

FMMCell(center::SVector{3,Float64}, half_width::Float64) =
    FMMCell(center, half_width, Int[], FMMCell[], 0.0, zero(SVector{3,Float64}), zero(SVector{3,Float64}), zero(SMatrix{3,3,Float64}), zero(SVector{3,Float64}), FMMCell[], true)

function bounding_box(pos::Matrix{Float64})
    min_corner = minimum(pos, dims = 2)[:]
    max_corner = maximum(pos, dims = 2)[:]
    center = (min_corner .+ max_corner) ./ 2.0
    half_width = maximum(max_corner .- min_corner) / 2.0
    return SVector{3,Float64}(center...), half_width
end

function cell_index_for_point(point::SVector{3,Float64}, center::SVector{3,Float64})
    idx = 0
    for i in 1:3
        if point[i] >= center[i]
            idx += 1 << (i - 1)
        end
    end
    return idx
end

function compute_cell_moments!(cell::FMMCell, pos::Matrix{Float64}, masses::Vector{Float64})
    if isempty(cell.particle_ids)
        cell.mass = 0.0
        cell.center_of_mass = zero(SVector{3,Float64})
        cell.dipole = zero(SVector{3,Float64})
        cell.quadrupole = zero(SMatrix{3,3,Float64})
        cell.is_leaf = true
        return
    end

    cell.mass = sum(masses[id] for id in cell.particle_ids)
    if cell.mass == 0.0
        cell.center_of_mass = zero(SVector{3,Float64})
        cell.dipole = zero(SVector{3,Float64})
        cell.quadrupole = zero(SMatrix{3,3,Float64})
        cell.is_leaf = true
        return
    end

    weighted = zero(SVector{3,Float64})
    for id in cell.particle_ids
        p = SVector{3,Float64}(pos[1, id], pos[2, id], pos[3, id])
        weighted += p * masses[id]
    end
    cell.center_of_mass = weighted / cell.mass

    dipole = zero(SVector{3,Float64})
    quadrupole = zero(SMatrix{3,3,Float64})
    for id in cell.particle_ids
        p = SVector{3,Float64}(pos[1, id], pos[2, id], pos[3, id])
        rel = p - cell.center_of_mass
        dipole += rel * masses[id]
        outer = SMatrix{3,3,Float64}(rel[1]*rel[1], rel[1]*rel[2], rel[1]*rel[3],
                                    rel[2]*rel[1], rel[2]*rel[2], rel[2]*rel[3],
                                    rel[3]*rel[1], rel[3]*rel[2], rel[3]*rel[3])
        I3 = SMatrix{3,3,Float64}(1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0)
        quadrupole += masses[id] * (3.0 * outer - dot(rel, rel) * I3)
    end
    cell.dipole = dipole
    cell.quadrupole = quadrupole
end

function build_tree(pos::Matrix{Float64}, masses::Vector{Float64}; leaf_size::Int = 8)
    root_center, root_half = bounding_box(pos)
    root = FMMCell(root_center, root_half)
    root.particle_ids = collect(1:size(pos, 2))
    subdivide_cell!(root, pos, masses, leaf_size)
    return root
end

function subdivide_cell!(cell::FMMCell, pos::Matrix{Float64}, masses::Vector{Float64}, leaf_size::Int)
    compute_cell_moments!(cell, pos, masses)

    if length(cell.particle_ids) <= leaf_size || cell.half_width <= 1e-12
        cell.children = FMMCell[]
        cell.is_leaf = true
        return
    end

    child_half = cell.half_width / 2.0
    child_particles = [Int[] for _ in 1:8]

    for id in cell.particle_ids
        p = SVector{3,Float64}(pos[1, id], pos[2, id], pos[3, id])
        idx = cell_index_for_point(p, cell.center)
        push!(child_particles[idx + 1], id)
    end

    cell.children = FMMCell[]
    offsets = [
        (-1.0, -1.0, -1.0), (1.0, -1.0, -1.0), (-1.0, 1.0, -1.0), (1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),  (1.0, -1.0, 1.0),  (-1.0, 1.0, 1.0),  (1.0, 1.0, 1.0)
    ]

    for k in 1:8
        if isempty(child_particles[k])
            continue
        end
        offset = offsets[k]
        child_center = cell.center + SVector{3,Float64}(offset...) * child_half
        child = FMMCell(child_center, child_half)
        child.particle_ids = child_particles[k]
        subdivide_cell!(child, pos, masses, leaf_size)
        push!(cell.children, child)
    end

    cell.is_leaf = isempty(cell.children)
    if !cell.is_leaf
        cell.particle_ids = Int[]
    end
end

function direct_cell_interaction!(target_particle::Int, source_ids::Vector{Int},
                                 pos::Matrix{Float64}, masses::Vector{Float64}, acc::Vector{Float64})
    p_target = SVector{3,Float64}(pos[1, target_particle], pos[2, target_particle], pos[3, target_particle])
    for id in source_ids
        if id == target_particle
            continue
        end
        p_source = SVector{3,Float64}(pos[1, id], pos[2, id], pos[3, id])
        dx = p_target - p_source
        dist2 = dot(dx, dx)
        if dist2 <= 1e-12
            continue
        end
        dist = sqrt(dist2)
        acc .-= (FMM_G * masses[id] / dist^3) .* dx
    end
end

function is_ancestor_of(ancestor::FMMCell, candidate::FMMCell)
    if ancestor === candidate
        return true
    end
    for child in ancestor.children
        if is_ancestor_of(child, candidate)
            return true
        end
    end
    return false
end

function is_well_separated(a_center::SVector{3,Float64}, b_center::SVector{3,Float64},
                          a_half::Float64, b_half::Float64, theta::Float64)
    dx = a_center - b_center
    dist = norm(dx)
    if dist <= 1e-12
        return false
    end
    return max(a_half, b_half) / dist < theta
end

function multipole_to_local_contribution(source::FMMCell, target::FMMCell)
    dx = target.center_of_mass - source.center_of_mass
    dist2 = dot(dx, dx)
    if dist2 <= 1e-12
        return zero(SVector{3,Float64})
    end
    dist = sqrt(dist2)
    rhat = dx / dist
    accel = (FMM_G * source.mass / dist^3) * dx
    accel += FMM_G * ((3.0 * dot(source.dipole, rhat) * rhat - source.dipole) / dist^3)
    accel += FMM_G * ((source.quadrupole * dx) / dist^5)
    return -SVector{3,Float64}(accel[1], accel[2], accel[3])
end

function collect_interaction_list!(target::FMMCell, node::FMMCell, theta::Float64)
    if node === target || is_ancestor_of(target, node) || is_ancestor_of(node, target)
        return
    end

    if is_well_separated(target.center_of_mass, node.center_of_mass, target.half_width, node.half_width, theta)
        push!(target.interaction_list, node)
        return
    end

    if !isempty(node.children)
        for child in node.children
            collect_interaction_list!(target, child, theta)
        end
    end
end

function build_interaction_lists!(cell::FMMCell, root::FMMCell, theta::Float64)
    cell.interaction_list = FMMCell[]
    _collect_interaction_lists!(cell, root, theta)
    for child in cell.children
        build_interaction_lists!(child, root, theta)
    end
end

function _collect_interaction_lists!(target::FMMCell, node::FMMCell, theta::Float64)
    if node === target
        if !isempty(node.children)
            for child in node.children
                _collect_interaction_lists!(target, child, theta)
            end
        end
        return
    end

    if is_ancestor_of(target, node) || is_ancestor_of(node, target)
        return
    end

    if is_well_separated(target.center_of_mass, node.center_of_mass, target.half_width, node.half_width, theta)
        push!(target.interaction_list, node)
        return
    end

    if !isempty(node.children)
        for child in node.children
            _collect_interaction_lists!(target, child, theta)
        end
    end
end

function downward_pass!(cell::FMMCell)
    cell.local_expansion = zero(SVector{3,Float64})
    for other in cell.interaction_list
        cell.local_expansion += multipole_to_local_contribution(other, cell)
    end
    for child in cell.children
        downward_pass!(child)
    end
end

function far_field_multipole_acceleration!(target_particle::Int, cell::FMMCell,
                                         pos::Matrix{Float64}, acc::Vector{Float64})
    if cell.mass == 0.0
        return
    end

    p_target = SVector{3,Float64}(pos[1, target_particle], pos[2, target_particle], pos[3, target_particle])
    dx = p_target - cell.center_of_mass
    dist2 = dot(dx, dx)
    if dist2 <= 1e-12
        return
    end
    dist = sqrt(dist2)
    rhat = dx / dist

    monopole = (FMM_G * cell.mass / dist^3) * dx
    dipole_term = (3.0 * dot(cell.dipole, rhat) * rhat - cell.dipole) / dist^3
    quadrupole_term = (cell.quadrupole * dx) / dist^5
    acc .-= monopole .+ FMM_G .* dipole_term .+ FMM_G .* quadrupole_term
end

function accumulate_tree_acceleration!(target_particle::Int, cell::FMMCell,
                                      pos::Matrix{Float64}, masses::Vector{Float64},
                                      theta::Float64, acc::Vector{Float64})
    if cell.mass == 0.0
        return
    end

    p_target = SVector{3,Float64}(pos[1, target_particle], pos[2, target_particle], pos[3, target_particle])
    if cell.is_leaf
        acc .+= cell.local_expansion
        direct_cell_interaction!(target_particle, cell.particle_ids, pos, masses, acc)
        return
    end

    target_in_cell = all(abs(p_target[d] - cell.center[d]) <= cell.half_width for d in 1:3)
    if !target_in_cell && is_well_separated(p_target, cell.center_of_mass, cell.half_width, cell.half_width, theta)
        acc .+= cell.local_expansion
        return
    end

    for child in cell.children
        accumulate_tree_acceleration!(target_particle, child, pos, masses, theta, acc)
    end
end

function compute_tree_accelerations(pos::Matrix{Float64}, masses::Vector{Float64}; theta::Float64 = 0.5, leaf_size::Int = 8)
    tree = build_tree(pos, masses; leaf_size = leaf_size)
    build_interaction_lists!(tree, tree, theta)
    downward_pass!(tree)
    accs = zeros(3, size(pos, 2))
    for i in 1:size(pos, 2)
        acc = zeros(3)
        accumulate_tree_acceleration!(i, tree, pos, masses, theta, acc)
        accs[:, i] .= acc
    end
    return accs
end

function direct_accelerations(pos::Matrix{Float64}, masses::Vector{Float64})
    accs = zeros(3, size(pos, 2))
    for i in 1:size(pos, 2)
        acc = zeros(3)
        for j in 1:size(pos, 2)
            if i == j
                continue
            end
            dx = SVector{3,Float64}(pos[1, i], pos[2, i], pos[3, i]) -
                 SVector{3,Float64}(pos[1, j], pos[2, j], pos[3, j])
            dist2 = dot(dx, dx)
            if dist2 <= 1e-12
                continue
            end
            dist = sqrt(dist2)
            acc .-= (FMM_G * masses[j] / dist^3) .* dx
        end
        accs[:, i] .= acc
    end
    return accs
end

function validate_fmm_against_direct(; n_particles::Int = 16, leaf_size::Int = 4)
    pos = rand(3, n_particles)
    masses = ones(n_particles) ./ n_particles
    direct = direct_accelerations(pos, masses)
    approx = compute_tree_accelerations(pos, masses; theta = 0.5, leaf_size = leaf_size)
    err = maximum(abs.(approx .- direct))
    return err, direct, approx
end

function full_fmm_step!(pos::Matrix{Float64}, vel::Matrix{Float64}, masses::Vector{Float64}, Δt::Float64;
                       theta::Float64 = 0.5, leaf_size::Int = 8)
    accs = compute_tree_accelerations(pos, masses; theta = theta, leaf_size = leaf_size)
    vel .-= accs .* Δt
    pos .+= vel .* Δt
    return accs
end

function simulation_step!(pos::Matrix{Float64}, vel::Matrix{Float64}, masses::Vector{Float64}, Δt::Float64)
    full_fmm_step!(pos, vel, masses, Δt; theta = 0.5, leaf_size = 8)
end

function mpi_local_range(n_particles::Int, rank::Int, n_ranks::Int)
    base, remainder = divrem(n_particles, n_ranks)
    first_particle = rank * base + min(rank, remainder) + 1
    last_particle = first_particle + base - 1 + (rank < remainder)
    return first_particle:last_particle
end

function simulation_step_mpi!(pos::Matrix{Float64}, vel::Matrix{Float64}, masses::Vector{Float64},
                             Δt::Float64, comm, rank::Int, n_ranks::Int)
    accs = compute_tree_accelerations(pos, masses; theta = 0.5, leaf_size = 8)
    vel .-= accs .* Δt
    pos .+= vel .* Δt
    return nothing
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
num_steps = 200

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
