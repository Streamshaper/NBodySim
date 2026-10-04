using Printf

function write_log(algorithm_type::String, num_particles::Int, num_steps::Int,
                   simulation_time::Float64, encoding_time::Float64)
    stopwatch_file = joinpath("logs", "stopwatch.csv")
    mkpath(dirname(stopwatch_file))
    core_count = parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", string(Sys.CPU_THREADS)))
    node_count = parse(Int, get(ENV, "SLURM_JOB_NUM_NODES", "1"))

    open(stopwatch_file, "a+") do io
        if filesize(stopwatch_file) == 0
            println(io, "type,node_count,core_count,particle_count,steps,simulation_time,encoding_time")
        end
        simulation_time_string = @sprintf("%.2f", simulation_time)
        encoding_time_string = @sprintf("%.2f", encoding_time)

        println(io, "$algorithm_type,$node_count,$core_count,$num_particles,$num_steps,$simulation_time_string,$encoding_time_string")
    end
end
