include(joinpath(@__DIR__, "profile.jl"))
include(joinpath(@__DIR__, "verification.jl"))
include(joinpath(@__DIR__, "logging.jl"))
include(joinpath(@__DIR__, "video_encoding.jl"))
include(joinpath(@__DIR__, "data_storage.jl"))

function grav_acc(mass::Float64, dx::Float64, dy::Float64, dz::Float64,
                  interaction_strength::Float64, smoothing::Float64,
                  kernel::AbstractKernel = PlummerGravity(interaction_strength, smoothing))::NTuple{3, Float64}
    target = ParticleProperties(1.0, 0.0)
    source = ParticleProperties(mass, 0.0)
    return kernel_acceleration(kernel, target, source, dx, dy, dz)
end

function mpi_local_range(n_particles::Int, rank::Int, n_ranks::Int)
    base, remainder = divrem(n_particles, n_ranks)
    first_particle = rank * base + min(rank, remainder) + 1
    last_particle = first_particle + base - 1 + (rank < remainder)
    return first_particle:last_particle
end
