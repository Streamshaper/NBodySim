# video_encoding.jl
using CairoMakie
using Printf

function encode_video_and_log(frames::Vector{Matrix{Float64}}, masses::Vector{Float64}, 
                              num_particles::Int, num_steps::Int, fps::Real, 
                              simulation_time::Float64, algorithm_type::String, file_prefix::String)
    
    # Ensure the output directory exists
    out_dir = "output"
    mkpath(out_dir)

    # Define the full path
    video_filename = "$(file_prefix)-animation_$(num_particles)p_$(num_steps)s.mp4"
    out_file = joinpath(out_dir, video_filename)

    println("Setting up CairoMakie animation for $algorithm_type...")
    flush(stdout)

    fig = Figure(size = (1600, 800), backgroundcolor = :black, figure_padding = 0)

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
                 protrusions = (0, 0, 0, 0),
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

    min_mass = minimum(masses)
    max_mass = maximum(masses)
    
    if isapprox(min_mass, max_mass)
        marker_sizes = fill(1.5, length(masses))
    else
        mass_scale = cbrt.(masses ./ max_mass)
        marker_sizes = 1.5 .+ 9.0 .* mass_scale
    end

    # Draw the initial scatter plot into both axes
    for ax in axs
        scatter!(ax, x_obs, y_obs, z_obs, color = (:white, 0.5), markersize = marker_sizes)
    end

    total_frames = length(frames)
    vid_log_interval = max(1, total_frames ÷ 10)

    println("Starting video encoding to $out_file ...")
    flush(stdout)

    video_encoding_time = @elapsed begin
        record(fig, out_file, 1:total_frames; framerate = fps) do i
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

    # Logging to stopwatch.csv
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
        
        println(io, "$algorithm_type,$node_count,$core_count,$num_particles,$num_steps,$simulation_time_string,$encoding_time_string")
    end
end
