using HDF5

function save_simulation_hdf5(frames::Vector{Matrix{Float64}}, 
                              vel_frames::Vector{Matrix{Float64}}, 
                              masses::Vector{Float64}, 
                              profile, 
                              num_particles::Int, 
                              num_steps::Int,
                              algorithm_type::String)
    
    h5_filename = joinpath("output", "simulations", "nbodysim_$(algorithm_type)_$(num_particles)p_$(num_steps)s.h5")
    mkpath(dirname(h5_filename))
    
    println("Writing simulation data to $h5_filename...")
    
    h5open(h5_filename, "w") do h5file
        # Write Metadata Attributes
        HDF5.attributes(h5file)["interaction_strength"] = profile.interaction_strength
        HDF5.attributes(h5file)["smoothing"] = profile.smoothing
        HDF5.attributes(h5file)["timestep"] = profile.timestep
        HDF5.attributes(h5file)["num_particles"] = num_particles
        HDF5.attributes(h5file)["num_steps"] = num_steps
        
        # Write Static Data
        h5file["masses"] = masses
        
        num_saved_steps = length(frames)
        
        # Create datasets on disk
        d_pos = create_dataset(h5file, "positions", Float64, (3, num_particles, num_saved_steps), 
                               chunk=(3, num_particles, 1), compress=3)
        d_vel = create_dataset(h5file, "velocities", Float64, (3, num_particles, num_saved_steps), 
                               chunk=(3, num_particles, 1), compress=3)
        
        # Write data slice-by-slice from RAM to Disk
        for i in 1:num_saved_steps
            d_pos[:, :, i] = frames[i]
            d_vel[:, :, i] = vel_frames[i]
        end
    end
    
    println("Data write complete: Saved to $h5_filename.")
    flush(stdout)
    return h5_filename
end