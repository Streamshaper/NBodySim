using DrWatson
@quickactivate "AHRB"

using AbstractTrees, AdaptiveHierarchicalRegularBinning, BenchmarkTools, Random, DataFrames, ColorSchemes, Colors, ProgressMeter, CairoMakie

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

getmass(node::SpatialTree) = getcontext(node)[:mass]
getcom(node::SpatialTree)  = getcontext(node)[:com]

"""
    grav_acc(mass, r; ϵ = 0.02)
"""
function grav_acc(mass::Float64, r::Vector{Float64}; ϵ::Float64 = 0.02)
    G::Float64 = 1.0 # Standardized unit scale
    d2 = sum(r.^2) + ϵ^2
    return ((G * mass) / (d2^(1.5))) .* r
end

"""
    net_acc(pos, vel, mass, node, tree, θ, all_masses)
"""
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
# WARM-UP COMPILATION SPINNER
# ---------------------------------------------------------
function compile_with_spinner(func::Function, message::String)
    # ... (Keep your existing compile_with_spinner function exactly as is) ...
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

compile_with_spinner("Compiling physics functions...") do
    warm_pos, warm_vel, warm_mass = rand(3, 2), rand(3, 2), [0.5, 0.5]
    warm_tree = ahrb(warm_pos, 2, 1; ctxtype = NamedTuple{(:com, :mass), Tuple{Vector{Float64}, Float64}})
    
    # FIX: Initialize the context before running the simulation step!
    update_mass_com!(warm_tree, warm_mass)
    
    simulation_step!(warm_pos, warm_vel, warm_mass, warm_tree, 0.0005, 0.5)
end

# ---------------------------------------------------------
# MAIN SIMULATION
# ---------------------------------------------------------
frames = Vector{Matrix{Float64}}()
num_particles = 1000
num_steps = 1000

pos, vel, masses = rand_particles(num_particles)
push!(frames, copy(pos)) # Push initial state (t=0)

println("Starting Galaxy Simulation...")
flush(stdout)

log_interval = max(1, num_steps ÷ 10) # Log every 10%

for step in 1:num_steps
    tree = ahrb(pos, 10, 4; ctxtype = NamedTuple{(:com, :mass), Tuple{Vector{Float64}, Float64}})

    update_mass_com!(tree, masses)
    simulation_step!(pos, vel, masses, tree, 0.0005, 0.5)
    push!(frames, copy(pos))
    
    # Print progress and force write to SLURM logs
    if step % log_interval == 0
        percent = round(Int, (step / num_steps) * 100)
        println("Simulation progress: $step / $num_steps steps ($percent%)")
        flush(stdout)
    end
end

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

println("Saved $out_file successfully!")
flush(stdout)
