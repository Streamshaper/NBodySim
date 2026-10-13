# video_encoding.jl
using CairoMakie
using LinearAlgebra
using Statistics

"""
    compute_percentile_radius(frame, masses, percentile; mass_weighted=true)

Calculates the bounding radius from the center of mass enclosing `percentile` 
fraction of the system (0.0 to 1.0).
"""
function compute_percentile_radius(frame::Matrix{Float64}, masses::Vector{Float64}, 
                                   percentile::Float64; mass_weighted::Bool = true)
    total_mass = sum(masses)
    com = sum(frame .* masses', dims=2) ./ total_mass
    radii = [norm(frame[:, col] .- com) for col in 1:size(frame, 2)]

    if mass_weighted
        idx = sortperm(radii)
        cum_mass = cumsum(masses[idx]) ./ total_mass
        
        # Safely find first index exceeding percentile, falling back to last element
        match_idx = findfirst(>=(percentile), cum_mass)
        target_idx = isnothing(match_idx) ? idx[end] : idx[match_idx]
        return radii[target_idx]
    else
        return quantile(radii, percentile)
    end
end

# Render the saved positions as a 3D animation and record timing metadata for benchmarking.
function encode_video(frames::Vector{Matrix{Float64}}, masses::Vector{Float64},
                      num_particles::Int, num_steps::Int, fps::Real, file_prefix::String)
    # Output path setup.
    out_dir = "output"
    mkpath(out_dir)

    video_filename = "$(file_prefix)-animation_$(num_particles)p_$(num_steps)s.mp4"
    out_file = joinpath(out_dir, video_filename)

    println("Setting up CairoMakie animation for $file_prefix...")
    flush(stdout)

    # Resolution setup (1920x960) & dynamic marker scale factor relative to 1600x800 baseline
    fig_width, fig_height = 1920, 960
    scale_factor = fig_width / 1920.0

    fig = Figure(size = (fig_width, fig_height), backgroundcolor = :black, figure_padding = 0)

    # Ensure the frame data is valid before plotting.
    all(frame -> all(isfinite, frame), frames) ||
        error("Cannot encode video: simulation produced non-finite particle positions")
    plot_scale = maximum(maximum(abs, frame) for frame in frames)
    isfinite(plot_scale) && plot_scale > 0 || error("Cannot encode video: particle positions exceed finite plotting limits")

    # Normalized frames for plotting
    norm_init_frame = frames[1] ./ plot_scale
    norm_end_frame = frames[end] ./ plot_scale

    # Compute bounding radii (with a 1% padding margin)
    # Left view: 100th percentile at simulation end
    # Right view: 60th percentile at initial state
    r_left  = compute_percentile_radius(norm_end_frame, masses, 1.0; mass_weighted = true)
    r_right = compute_percentile_radius(norm_init_frame, masses, 0.6; mass_weighted = true)
    view_radii = [r_left, r_right]

    # Both views set to (60.0°, 35.0°) azimuth and elevation
    angle_az = deg2rad(60.0)
    angle_el = deg2rad(35.0)

    # Build the side-by-side plotting area with independent zoom bounds.
    axs = [Axis3(fig[1, col], 
                 azimuth = angle_az, 
                 elevation = angle_el,
                 limits = (-view_radii[col], view_radii[col], 
                           -view_radii[col], view_radii[col], 
                           -view_radii[col], view_radii[col]),
                 aspect = :data,
                 perspectiveness = 0.5,
                 protrusions = (0, 0, 0, 0),
                 backgroundcolor = :black)
           for col in 1:2]

    for ax in axs
        hidedecorations!(ax)
        hidespines!(ax)
    end

    # Baseline marker sizes for unit box (R = 1.0) scaled by figure resolution
    min_mass = minimum(masses)
    max_mass = maximum(masses)

    if isapprox(min_mass, max_mass)
        base_marker_sizes = fill(1.5 * scale_factor, length(masses))
    else
        mass_scale = cbrt.(masses ./ max_mass)
        base_marker_sizes = (1.5 .+ 10.5 .* mass_scale) .* scale_factor
    end

    # Both views scale marker size independently based on zoom ratio (1.0 / R_view)
    view_marker_sizes = [
        base_marker_sizes .* (1.0 / view_radii[col]) for col in 1:2
    ]

    # Create separate Observables per view to allow view-specific spherical masking
    x_obs = [Observable(Vector{Float64}(undef, num_particles)) for _ in 1:2]
    y_obs = [Observable(Vector{Float64}(undef, num_particles)) for _ in 1:2]
    z_obs = [Observable(Vector{Float64}(undef, num_particles)) for _ in 1:2]

    total_mass = sum(masses)

    # Helper function to apply spherical masking and update frame positions
    function update_frame!(frame::Matrix{Float64})
        norm_frame = frame ./ plot_scale
        com = sum(norm_frame .* masses', dims=2) ./ total_mass
        
        # Calculate distances from current center of mass
        dx = norm_frame[1, :] .- com[1]
        dy = norm_frame[2, :] .- com[2]
        dz = norm_frame[3, :] .- com[3]
        radii = sqrt.(dx.^2 .+ dy.^2 .+ dz.^2)

        for col in 1:2
            # Spherical cutoff mask: set out-of-bound particle coordinates to NaN for Makie to skip
            mask = radii .> view_radii[col]
            
            x_c = copy(norm_frame[1, :])
            y_c = copy(norm_frame[2, :])
            z_c = copy(norm_frame[3, :])

            x_c[mask] .= NaN
            y_c[mask] .= NaN
            z_c[mask] .= NaN

            x_obs[col][] = x_c
            y_obs[col][] = y_c
            z_obs[col][] = z_c
        end
    end

    # Attach scatter plots to each axis with their respective view-scaled marker sizes
    for col in 1:2
        scatter!(axs[col], x_obs[col], y_obs[col], z_obs[col], 
                 color = (:white, 0.5), markersize = view_marker_sizes[col])
    end

    # Initialize observables with first frame
    update_frame!(frames[1])

    total_frames = length(frames)
    vid_log_interval = max(1, total_frames ÷ 10)

    println("Starting video encoding to $out_file ...")
    flush(stdout)

    # Write the MP4 once the observed positions are ready.
    video_encoding_time = @elapsed begin
        record(fig, out_file, 1:total_frames; framerate = fps) do i
            update_frame!(frames[i])

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

    return video_encoding_time
end