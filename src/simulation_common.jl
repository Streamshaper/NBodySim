include(joinpath(@__DIR__, "profile.jl"))
include(joinpath(@__DIR__, "verification.jl"))
include(joinpath(@__DIR__, "logging.jl"))
include(joinpath(@__DIR__, "video_encoding.jl"))
include(joinpath(@__DIR__, "data_storage.jl"))

function grav_acc(mass::Float64, dx::Float64, dy::Float64, dz::Float64,
                  interaction_strength::Float64, smoothing::Float64)::NTuple{3, Float64}
    d2 = dx^2 + dy^2 + dz^2 + smoothing^2
    scale = (interaction_strength * mass) / (d2^1.5)
    return (scale * dx, scale * dy, scale * dz)
end

function mpi_local_range(n_particles::Int, rank::Int, n_ranks::Int)
    base, remainder = divrem(n_particles, n_ranks)
    first_particle = rank * base + min(rank, remainder) + 1
    last_particle = first_particle + base - 1 + (rank < remainder)
    return first_particle:last_particle
end
