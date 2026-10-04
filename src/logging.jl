using Printf

const STOPWATCH_V2_HEADER =
    "solver,kernel,integrator,node_count,core_count,particle_count,steps,simulation_time,encoding_time"

function write_log(solver::AbstractSolver, kernel::AbstractKernel, integrator::AbstractIntegrator,
                   num_particles::Int, num_steps::Int, simulation_time::Float64,
                   encoding_time::Float64)
    stopwatch_file = joinpath("logs", "stopwatch_v2.csv")
    mkpath(dirname(stopwatch_file))
    core_count = parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", string(Sys.CPU_THREADS)))
    node_count = parse(Int, get(ENV, "SLURM_JOB_NUM_NODES", "1"))

    open(stopwatch_file, "a+") do io
        if filesize(stopwatch_file) == 0
            println(io, STOPWATCH_V2_HEADER)
        else
            seekstart(io)
            existing_header = chomp(readline(io))
            existing_header == STOPWATCH_V2_HEADER ||
                error("Unexpected header in '$stopwatch_file'; refusing to append stopwatch v2 data")
        end
        simulation_time_string = @sprintf("%.2f", simulation_time)
        encoding_time_string = @sprintf("%.2f", encoding_time)

        println(io, "$(solver_name(solver)),$(kernel_name(kernel))," *
                    "$(integrator_name(integrator)),$node_count,$core_count," *
                    "$num_particles,$num_steps,$simulation_time_string,$encoding_time_string")
    end
end
