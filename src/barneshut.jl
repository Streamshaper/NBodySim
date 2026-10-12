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
getcharge(node::SpatialTree) = getcontext(node)[:charge]
getchargecom(node::SpatialTree) = getcontext(node)[:charge_com]
getdipole(node::SpatialTree) = getcontext(node)[:dipole]

function coulomb_cell_acceleration(target_charge::Float64, target_mass::Float64,
                                  charge_total::Float64, dipole::Vector{Float64},
                                  tx::Float64, ty::Float64, tz::Float64,
                                  interaction_strength::Float64)::NTuple{3, Float64}
    r2 = tx^2 + ty^2 + tz^2
    r2 <= 0.0 && return (0.0, 0.0, 0.0)
    r = sqrt(r2)
    inv_r3 = 1.0 / (r2 * r)
    field_x = interaction_strength * charge_total * tx * inv_r3
    field_y = interaction_strength * charge_total * ty * inv_r3
    field_z = interaction_strength * charge_total * tz * inv_r3

    if !all(iszero, dipole)
        d_dot_r = dipole[1] * tx + dipole[2] * ty + dipole[3] * tz
        dipole_scale = interaction_strength / (r^5)
        field_x += dipole_scale * (3.0 * d_dot_r * tx - dipole[1] * r2)
        field_y += dipole_scale * (3.0 * d_dot_r * ty - dipole[2] * r2)
        field_z += dipole_scale * (3.0 * d_dot_r * tz - dipole[3] * r2)
    end

    return (target_charge * field_x / target_mass,
            target_charge * field_y / target_mass,
            target_charge * field_z / target_mass)
end

# Barnes-Hut acceleration: exact for near leaves, approximate for far cells.
function net_acc(px::Float64, py::Float64, pz::Float64, node::SpatialTree,
                 tree::SpatialTree, θ::Float64, all_masses::Vector{Float64},
                 all_charges::Vector{Float64}, target_charge::Float64,
                 target_mass::Float64, kernel::AbstractKernel = PlummerGravity(1.0, 0.0))::NTuple{3, Float64}
    if !(kernel isa Coulomb)
        s = sidelength(node)
        com = getcom(node)
        rx = com[1] - px
        ry = com[2] - py
        rz = com[3] - pz
        dist_sq = rx^2 + ry^2 + rz^2
        dist_sq <= 0.0 && return (0.0, 0.0, 0.0)

        if isleaf(node)
            pp = points(node)
            idx = range(node)

            acc_x = acc_y = acc_z = 0.0
            for i in axes(pp, 2)
                idx_orig = tree.info.perm[idx[i]]
                dx = pp[1, i] - px
                dy = pp[2, i] - py
                dz = pp[3, i] - pz
                d_sq = dx^2 + dy^2 + dz^2

                if d_sq > 1e-12
                    source = ParticleProperties(all_masses[idx_orig], 0.0)
                    grav_x, grav_y, grav_z = kernel_acceleration(kernel, ParticleProperties(1.0, 0.0), source, dx, dy, dz)
                    acc_x += grav_x
                    acc_y += grav_y
                    acc_z += grav_z
                end
            end
            return (acc_x, acc_y, acc_z)
        end

        if s / sqrt(dist_sq) < θ
            target = ParticleProperties(target_mass, 0.0)
            source = ParticleProperties(getmass(node), 0.0)
            return kernel_acceleration(kernel, target, source, rx, ry, rz)
        end

        acc_x = acc_y = acc_z = 0.0
        for child in children(node)
            cx, cy, cz = net_acc(px, py, pz, child, tree, θ, all_masses,
                                 all_charges, target_charge, target_mass, kernel)
            acc_x += cx
            acc_y += cy
            acc_z += cz
        end
        return (acc_x, acc_y, acc_z)
    end
    s = sidelength(node)
    com = getcom(node)
    charge_center = getchargecom(node)

    rx = com[1] - px
    ry = com[2] - py
    rz = com[3] - pz
    dist_sq = rx^2 + ry^2 + rz^2

    if isleaf(node)
        pp = points(node)
        idx = range(node)

        acc_x = acc_y = acc_z = 0.0

        for i in axes(pp, 2)
            idx_orig = tree.info.perm[idx[i]]

            dx = pp[1, i] - px
            dy = pp[2, i] - py
            dz = pp[3, i] - pz
            d_sq = dx^2 + dy^2 + dz^2

            if d_sq > 1e-12
                target = ParticleProperties(target_mass, target_charge)
                source = ParticleProperties(all_masses[idx_orig], all_charges[idx_orig])
                grav_x, grav_y, grav_z = kernel_acceleration(kernel, target, source, dx, dy, dz)
                acc_x += grav_x
                acc_y += grav_y
                acc_z += grav_z
            end
        end
        return (acc_x, acc_y, acc_z)

    elseif s / sqrt(dist_sq) < θ
        if kernel isa Coulomb
            tx = px - charge_center[1]
            ty = py - charge_center[2]
            tz = pz - charge_center[3]
            return coulomb_cell_acceleration(target_charge, target_mass, getcharge(node),
                                             getdipole(node), tx, ty, tz,
                                             kernel.interaction_strength)
        end
        target = ParticleProperties(target_mass, target_charge)
        source = ParticleProperties(getmass(node), getcharge(node))
        return kernel_acceleration(kernel, target, source, rx, ry, rz)

    else
        acc_x = acc_y = acc_z = 0.0

        for child in children(node)
            cx, cy, cz = net_acc(px, py, pz, child, tree, θ, all_masses,
                                 all_charges, target_charge, target_mass, kernel)
            acc_x += cx
            acc_y += cy
            acc_z += cz
        end

        return (acc_x, acc_y, acc_z)
    end
end

function simulation_step_barneshut_mpi!(pos::Matrix{Float64}, vel::Matrix{Float64}, masses::Vector{Float64},
                                        charges::Vector{Float64}, tree::SpatialTree, profile::SimulationProfile,
                                        comm, rank::Int, n_ranks::Int,
                                        local_pos::Matrix{Float64}, local_vel::Matrix{Float64},
                                        gathered_pos::Matrix{Float64}, gathered_vel::Matrix{Float64},
                                        gather_counts::Vector{Int})
    
    local_range = mpi_local_range(size(pos, 2), rank, n_ranks)

    fill!(local_pos, 0.0)
    fill!(local_vel, 0.0)

    Threads.@threads :dynamic for idx in local_range
        i = tree.info.perm[idx]
        px = pos[1, i]
        py = pos[2, i]
        pz = pos[3, i]

        acc_x, acc_y, acc_z = net_acc(px, py, pz, tree, tree,
                                      profile.barnes_hut_opening_angle, masses,
                                      charges, charges[i], masses[i], profile.kernel)
        integrate_particle!(profile.integrator, local_pos, local_vel, idx, pos, vel, i,
                            (acc_x, acc_y, acc_z), profile.timestep)
    end

    MPI.Allgatherv!(view(local_pos, :, local_range), MPI.VBuffer(gathered_pos, gather_counts), comm)
    MPI.Allgatherv!(view(local_vel, :, local_range), MPI.VBuffer(gathered_vel, gather_counts), comm)

    for idx in axes(tree.info.perm, 1)
        i = tree.info.perm[idx]
        pos[1, i] = gathered_pos[1, idx]
        pos[2, i] = gathered_pos[2, idx]
        pos[3, i] = gathered_pos[3, idx]
        vel[1, i] = gathered_vel[1, idx]
        vel[2, i] = gathered_vel[2, idx]
        vel[3, i] = gathered_vel[3, idx]
    end
end

# Refresh aggregate mass and center of mass for each tree node without intermediate allocations.
function update_mass_com!(tree::SpatialTree, masses::Vector{Float64}, charges::Vector{Float64}=zeros(length(masses));
                          kernel::AbstractKernel = PlummerGravity(1.0, 0.0))
    if !(kernel isa Coulomb)
        foreach(PostOrderDFS(tree)) do node
            if isleaf(node)
                idx_range = range(node)
                node_mass = 0.0
                cx = cy = cz = 0.0
                np = points(node)
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
                setcontext!(node, (; com = [cx, cy, cz], mass = node_mass,
                                    charge = 0.0, charge_com = [cx, cy, cz],
                                    dipole = [0.0, 0.0, 0.0]))
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
                setcontext!(node, (; com = [cx, cy, cz], mass = node_mass,
                                    charge = 0.0, charge_com = [cx, cy, cz],
                                    dipole = [0.0, 0.0, 0.0]))
            end
        end
        return nothing
    end

    foreach(PostOrderDFS(tree)) do node
        if isleaf(node)
            idx_range = range(node)
            node_mass = 0.0
            node_charge = 0.0
            cx = cy = cz = 0.0
            qx = qy = qz = 0.0
            dipole_x = dipole_y = dipole_z = 0.0

            np = points(node)
            for i in axes(np, 2)
                idx_orig = tree.info.perm[idx_range[i]]
                m = masses[idx_orig]
                q = charges[idx_orig]

                node_mass += m
                node_charge += q
                cx += np[1, i] * m
                cy += np[2, i] * m
                cz += np[3, i] * m
                qx += np[1, i] * q
                qy += np[2, i] * q
                qz += np[3, i] * q
            end

            if node_mass > 0
                cx /= node_mass
                cy /= node_mass
                cz /= node_mass
            end
            if abs(node_charge) > 0
                charge_center_x = qx / node_charge
                charge_center_y = qy / node_charge
                charge_center_z = qz / node_charge
                for i in axes(np, 2)
                    idx_orig = tree.info.perm[idx_range[i]]
                    q = charges[idx_orig]
                    dipole_x += q * (np[1, i] - charge_center_x)
                    dipole_y += q * (np[2, i] - charge_center_y)
                    dipole_z += q * (np[3, i] - charge_center_z)
                end
                qx = charge_center_x
                qy = charge_center_y
                qz = charge_center_z
            else
                qx, qy, qz = cx, cy, cz
            end
            setcontext!(node, (; com = [cx, cy, cz], mass = node_mass,
                                charge = node_charge, charge_com = [qx, qy, qz],
                                dipole = [dipole_x, dipole_y, dipole_z]))
        else
            cx = cy = cz = 0.0
            qx = qy = qz = 0.0
            node_mass = 0.0
            node_charge = 0.0
            for child in children(node)
                ctx = getcontext(child)
                ccom = ctx[:com]
                cmass = ctx[:mass]
                ccharge = ctx[:charge]
                ccharge_com = ctx[:charge_com]

                cx += ccom[1] * cmass
                cy += ccom[2] * cmass
                cz += ccom[3] * cmass
                qx += ccharge_com[1] * ccharge
                qy += ccharge_com[2] * ccharge
                qz += ccharge_com[3] * ccharge
                node_mass += cmass
                node_charge += ccharge
            end

            if node_mass > 0
                cx /= node_mass
                cy /= node_mass
                cz /= node_mass
            end
            if abs(node_charge) > 0
                charge_center_x = qx / node_charge
                charge_center_y = qy / node_charge
                charge_center_z = qz / node_charge
            else
                charge_center_x = cx
                charge_center_y = cy
                charge_center_z = cz
            end

            dipole_x = dipole_y = dipole_z = 0.0
            for child in children(node)
                ctx = getcontext(child)
                ccharge = ctx[:charge]
                ccharge_com = ctx[:charge_com]
                cdipole = ctx[:dipole]
                if abs(ccharge) > 0
                    dipole_x += ccharge * (ccharge_com[1] - charge_center_x) + cdipole[1]
                    dipole_y += ccharge * (ccharge_com[2] - charge_center_y) + cdipole[2]
                    dipole_z += ccharge * (ccharge_com[3] - charge_center_z) + cdipole[3]
                end
            end

            setcontext!(node, (; com = [cx, cy, cz], mass = node_mass,
                                charge = node_charge, charge_com = [charge_center_x, charge_center_y, charge_center_z],
                                dipole = [dipole_x, dipole_y, dipole_z]))
        end
    end
    return nothing
end

function run_barneshut_simulation(profile_path::AbstractString="profiles/default.toml",
                                num_steps_override::Union{Nothing,Int}=nothing;
                                solver_override::BarnesHutSolver=BarnesHutSolver())
    profile_file = isabspath(profile_path) ? profile_path : joinpath(dirname(@__DIR__), profile_path)
    profile = load_profile(profile_file; solver_override)
    require_solver_support(BarnesHutSolver(), profile.kernel, profile.integrator)

    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    n_ranks = MPI.Comm_size(comm)

    frames = Vector{Matrix{Float64}}()
    vel_frames = Vector{Matrix{Float64}}()
    num_particles = size(profile.positions, 2)
    num_steps = num_steps_override === nothing ? profile.num_steps : Int(num_steps_override)

    verification_baseline = profile.verification_enabled ?
        verification_reference(profile.positions, profile.velocities,
                               profile.masses, profile.charges,
                               profile.kernel,
                               profile.verification_energy_max_particles) : nothing
    verification_rows = rank == 0 && profile.verification_enabled ?
        [(0, 0.0, verification_metrics(profile.positions, profile.velocities,
                                        profile.masses, profile.charges,
                                        profile.kernel, verification_baseline,
                                        profile.verification_energy_max_particles))] : nothing

    pos, vel, masses, charges = rank == 0 ?
        (copy(profile.positions), copy(profile.velocities), copy(profile.masses), copy(profile.charges)) :
        (zeros(3, num_particles), zeros(3, num_particles), zeros(num_particles), zeros(num_particles))
    MPI.Bcast!(pos, 0, comm)
    MPI.Bcast!(vel, 0, comm)
    MPI.Bcast!(masses, 0, comm)
    MPI.Bcast!(charges, 0, comm)

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
    gathered_pos_buffer = zeros(3, num_particles)
    gathered_vel_buffer = zeros(3, num_particles)
    gather_counts = [3 * length(mpi_local_range(num_particles, r, n_ranks)) for r in 0:(n_ranks - 1)]

    simulation_time = @elapsed begin
        for step in 1:num_steps
            tree = ahrb(pos, 10, 4; ctxtype = NamedTuple{(:com, :mass, :charge, :charge_com, :dipole),
                Tuple{Vector{Float64}, Float64, Float64, Vector{Float64}, Vector{Float64}}
            })

            update_mass_com!(tree, masses, charges; kernel = profile.kernel)
            simulation_step_barneshut_mpi!(pos, vel, masses, charges, tree, profile, comm, rank, n_ranks,
                                           local_pos_buffer, local_vel_buffer, gathered_pos_buffer,
                                           gathered_vel_buffer, gather_counts)
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
                                              verification_metrics(pos, vel, masses, charges,
                                                                   profile.kernel, verification_baseline,
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
            verification_file = verification_log_path()
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

    encoding_time = 0.0
    if rank == 0 && profile.video_encoding_enabled
        encoding_time = encode_video(frames, masses, num_particles, num_steps,
                                     round(Int, profile.fps), "barneshut")
    end

    if rank == 0 && profile.logging_enabled
        write_log(BarnesHutSolver(), profile.kernel, profile.integrator,
                  num_particles, num_steps, simulation_time, encoding_time)
    end

    if rank == 0 && profile.store_data
        save_simulation_hdf5(frames, vel_frames, masses, charges, profile,
                             num_particles, num_steps, "B-H")
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
