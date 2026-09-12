using DrWatson
@quickactivate "AHRB"

using BenchmarkTools, Random, DataFrames, ColorSchemes, Colors, ProgressMeter, CairoMakie
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
FastMultipole.body_to_multipole!(system::GravitationalSystem, args...) =
FastMultipole.body_to_multipole!(Point{Source}, system, args...; scale_strength = -1.0)
FastMultipole.get_previous_influence(::GravitationalSystem, i) = nothing

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
                gradient -= SVector{3}(dx, dy, dz) * source_strength /
                            (4π * r2 * r)
            end
        end
        
        # FIXED: Removed the 'switch' argument. 
        # set_gradient! only needs the buffer, index, and value.
        GS && FastMultipole.set_gradient!(target_buffer, j_target, gradient)
    end
end

function FastMultipole.buffer_to_target_system!(target_system::GravitationalSystem, i_target,
                                                switch::FastMultipole.DerivativesSwitch{PS,GS,HS},
                                                target_buffer, i_buffer) where {PS,GS,HS}
    gradient = GS ? FastMultipole.get_gradient(target_buffer, switch, i_buffer) : zero(SVector{3,eltype(target_system)})
    target_system.potential[5:7, i_target] .= gradient
end

function simulation_step!(pos::Matrix{Float64}, vel::Matrix{Float64}, masses::Vector{Float64}, Δt::Float64)
    system = GravitationalSystem(pos, masses)
    fmm!(system; gradient = true)
    accs = @view system.potential[5:7, :]
    vel .+= accs .* Δt
    pos .+= vel .* Δt
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
    simulation_step!(warm_pos, warm_vel, warm_mass, 0.0005)
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
    simulation_step!(pos, vel, masses, 0.0005)
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
