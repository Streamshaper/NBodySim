using DrWatson
@quickactivate "AHRB"

using AbstractTrees, AdaptiveHierarchicalRegularBinning, BenchmarkTools, Random, DataFrames, ColorSchemes, Colors, ProgressMeter, CairoMakie
using Printf
using MPI
include(joinpath(@__DIR__, "profile.jl"))
include(joinpath(@__DIR__, "verification.jl"))
include(joinpath(@__DIR__, "video_encoding.jl"))
include(joinpath(@__DIR__, "data_storage.jl"))

getmass(node::SpatialTree) = getcontext(node)[:mass]
getcom(node::SpatialTree)  = getcontext(node)[:com]

function grav_acc(mass::Float64, r::Vector{Float64}, interaction_strength::Float64, smoothing::Float64)
    d2 = sum(r.^2) + smoothing^2
    return ((interaction_strength * mass) / (d2^(1.5))) .* r
end

function net_acc(pos::Vector{Float64}, vel::Vector{Float64}, mass::Float64, node::SpatialTree,
                 tree::SpatialTree, θ::Float64, all_masses::Vector{Float64},
                 interaction_strength::Float64, smoothing::Float64)
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
                acc .+= grav_acc(all_masses[idx_orig], r_vec, interaction_strength, smoothing)
            end
        end
        return acc
    elseif s / √(sum(r.^2)) < θ
        return grav_acc(getmass(node), r, interaction_strength, smoothing)
    else
        return sum(net_acc(pos, vel, mass, child, tree, θ, all_masses,
                           interaction_strength, smoothing) for child in children(node))
    end
end

function mpi_local_range(n_particles::Int, rank::Int, n_ranks::Int)
    base, remainder = divrem(n_particles, n_ranks)
    first_particle = rank * base + min(rank, remainder) + 1
    last_particle = first_particle + base - 1 + (rank < remainder)
    return first_particle:last_particle
end

function simulation_step_mpi!(pos::Matrix{Float64}, vel::Matrix{Float64}, masses::Vector{Float64},
                             tree::SpatialTree, profile::SimulationProfile, comm, rank::Int, n_ranks::Int)
    local_pos = zeros(size(pos))
    local_vel = zeros(size(vel))
    local_range = mpi_local_range(size(pos, 2), rank, n_ranks)

    Threads.@threads for i in local_range
        acc = net_acc(pos[:, i], vel[:, i], masses[i], tree, tree,
                  profile.barnes_hut_opening_angle, masses,
                  profile.interaction_strength, profile.smoothing)
        local_vel[:, i] .= vel[:, i] .+ acc .* profile.timestep
        local_pos[:, i] .= pos[:, i] .+ local_vel[:, i] .* profile.timestep
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
vel_frames = Vector{Matrix{Float64}}()
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

if MPI_RANK == 0
    if profile.video_encoding_enabled || profile.store_data
        push!(frames, copy(pos)) 
    end
    if profile.store_data
        push!(vel_frames, copy(vel)) # Track velocities for HDF5
    end
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
        simulation_step_mpi!(pos, vel, masses, tree, profile, MPI_COMM, MPI_RANK, MPI_SIZE)
        if MPI_RANK == 0 
            if profile.video_encoding_enabled || profile.store_data
                push!(frames, copy(pos))
            end
            if profile.store_data
                push!(vel_frames, copy(vel))
            end
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
    if profile.verification_enabled
        verification_file = joinpath("logs", "verification_barnes_hut_$(num_particles)p_$(num_steps)s.csv")
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
    encode_video_and_log(frames, masses, num_particles, num_steps, round(Int, profile.fps), simulation_time, "B_H", "barneshut")
end

if MPI_RANK == 0 && profile.store_data
    save_simulation_hdf5(frames, vel_frames, masses, profile, num_particles, num_steps, "B-H")
end

MPI.Barrier(MPI_COMM)
MPI.Finalize()
