using DrWatson
@quickactivate "AHRB"

using AbstractTrees, AdaptiveHierarchicalRegularBinning, BenchmarkTools, Random, DataFrames, ColorSchemes, Colors, ProgressMeter, CairoMakie
using Printf
using MPI

if !@isdefined __NBodySimCommonLoaded
    include(joinpath(@__DIR__, "simulation_common.jl"))
    global __NBodySimCommonLoaded = true
end

# Cached aggregate mass and center-of-mass for each tree node.
getmass(node::SpatialTree) = getcontext(node)[:mass]
getcom(node::SpatialTree)  = getcontext(node)[:com]

# Barnes-Hut acceleration: exact for near leaves, approximate for far cells.
function net_acc(px::Float64, py::Float64, pz::Float64, node::SpatialTree,
                 tree::SpatialTree, θ::Float64, all_masses::Vector{Float64},
                 interaction_strength::Float64, smoothing::Float64)::NTuple{3, Float64}
    
    s = sidelength(node)
    com = getcom(node) # AHRB returns a Vector{Float64} here
    
    rx = com[1] - px
    ry = com[2] - py
    rz = com[3] - pz
    dist_sq = rx^2 + ry^2 + rz^2

    if isleaf(node)
        pp  = points(node)
        idx = range(node)
        
        acc_x = acc_y = acc_z = 0.0

        for i in axes(pp, 2)
            idx_orig = tree.info.perm[idx[i]]
            
            dx = pp[1, i] - px
            dy = pp[2, i] - py
            dz = pp[3, i] - pz
            d_sq = dx^2 + dy^2 + dz^2

            # Skip self-interaction
            if d_sq > 1e-12
                grav_x, grav_y, grav_z = grav_acc(all_masses[idx_orig], dx, dy, dz,
                                                  interaction_strength, smoothing)
                acc_x += grav_x
                acc_y += grav_y
                acc_z += grav_z
            end
        end
        return (acc_x, acc_y, acc_z)
        
    elseif s / sqrt(dist_sq) < θ
        return grav_acc(getmass(node), rx, ry, rz, interaction_strength, smoothing)
        
    else
        acc_x = acc_y = acc_z = 0.0
        
        for child in children(node)
            cx, cy, cz = net_acc(px, py, pz, child, tree, θ, all_masses,
                                 interaction_strength, smoothing)
            acc_x += cx
            acc_y += cy
            acc_z += cz
        end
        
        return (acc_x, acc_y, acc_z)
    end
end

function simulation_step_barneshut_mpi!(pos::Matrix{Float64}, vel::Matrix{Float64}, masses::Vector{Float64},
                                        tree::SpatialTree, profile::SimulationProfile, comm, rank::Int, n_ranks::Int,
                                        local_pos::Matrix{Float64}, local_vel::Matrix{Float64})
    
    local_range = mpi_local_range(size(pos, 2), rank, n_ranks)

    fill!(local_pos, 0.0)
    fill!(local_vel, 0.0)

    Threads.@threads :dynamic for idx in local_range
        # Maximum Cache Efficiency: Use AHRB's spatial mapping
        i = tree.info.perm[idx]
        
        px = pos[1, i]
        py = pos[2, i]
        pz = pos[3, i]
        
        acc_x, acc_y, acc_z = net_acc(px, py, pz, tree, tree,
                                      profile.barnes_hut_opening_angle, masses,
                                      profile.interaction_strength, profile.smoothing)
        
        vx = vel[1, i] + acc_x * profile.timestep
        vy = vel[2, i] + acc_y * profile.timestep
        vz = vel[3, i] + acc_z * profile.timestep
        
        local_vel[1, i] = vx
        local_vel[2, i] = vy
        local_vel[3, i] = vz
        
        local_pos[1, i] = px + vx * profile.timestep
        local_pos[2, i] = py + vy * profile.timestep
        local_pos[3, i] = pz + vz * profile.timestep
    end

    MPI.Allreduce!(local_pos, pos, +, comm)
    MPI.Allreduce!(local_vel, vel, +, comm)
end

# Refresh aggregate mass and center of mass for each tree node without intermediate allocations.
function update_mass_com!(tree::SpatialTree, masses::Vector{Float64})
    foreach(PostOrderDFS(tree)) do node
        if isleaf(node)
            idx_range = range(node)
            node_mass = 0.0
            cx = cy = cz = 0.0
            
            np = points(node)
            # Allocation-free manual loop
            for i in axes(np, 2)
                idx_orig = tree.info.perm[idx_range[i]]
                m = masses[idx_orig]
                
                node_mass += m
                cx += np[1, i] * m
                cy += np[2, i] * m
                cz += np[3, i] * m
            end
            
            if node_mass > 0
                cx /= node_mass
                cy /= node_mass
                cz /= node_mass
            end
            setcontext!(node, (; com = [cx, cy, cz], mass = node_mass))
        else
            cx = cy = cz = 0.0
            node_mass = 0.0
            for child in children(node)
                ctx = getcontext(child)
                ccom = ctx[:com]
                cmass = ctx[:mass]
                
                cx += ccom[1] * cmass
                cy += ccom[2] * cmass
                cz += ccom[3] * cmass
                node_mass += cmass
            end
            
            if node_mass > 0
                cx /= node_mass
                cy /= node_mass
                cz /= node_mass
            end
            setcontext!(node, (; com = [cx, cy, cz], mass = node_mass))
        end
    end
end

function run_barneshut_simulation(profile_path::AbstractString="profiles/default.toml",
                                num_steps_override::Union{Nothing,Int}=nothing)
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    n_ranks = MPI.Comm_size(comm)

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

    local_pos_buffer = zeros(3, num_particles)
    local_vel_buffer = zeros(3, num_particles)

    simulation_time = @elapsed begin
        for step in 1:num_steps
            tree = ahrb(pos, 10, 4; ctxtype = NamedTuple{(:com, :mass), Tuple{Vector{Float64}, Float64}})

            update_mass_com!(tree, masses)
            simulation_step_barneshut_mpi!(pos, vel, masses, tree, profile, comm, rank, n_ranks, local_pos_buffer, local_vel_buffer)
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

    if rank == 0 && profile.video_encoding_enabled
        encode_video_and_log(frames, masses, num_particles, num_steps, round(Int, profile.fps), simulation_time, "B_H", "barneshut")
    end

    if rank == 0 && profile.store_data
        save_simulation_hdf5(frames, vel_frames, masses, profile, num_particles, num_steps, "B-H")
    end

    MPI.Barrier(comm)
    MPI.Finalize()
    return nothing
end

function main_barneshut(args::Vector{String}=ARGS)
    profile_path = length(args) >= 1 ? args[1] : "profiles/default.toml"
    num_steps = length(args) >= 2 ? parse(Int, args[2]) : nothing
    run_barneshut_simulation(profile_path, num_steps)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_barneshut(ARGS)
end
