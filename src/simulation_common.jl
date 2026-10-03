include(joinpath(@__DIR__, "profile.jl"))
include(joinpath(@__DIR__, "verification.jl"))
include(joinpath(@__DIR__, "video_encoding.jl"))
include(joinpath(@__DIR__, "data_storage.jl"))

function grav_acc(mass::Float64, r::Vector{Float64}, interaction_strength::Float64, smoothing::Float64)
    d2 = sum(r.^2) + smoothing^2
    return ((interaction_strength * mass) / (d2^(1.5))) .* r
end

function mpi_local_range(n_particles::Int, rank::Int, n_ranks::Int)
    base, remainder = divrem(n_particles, n_ranks)
    first_particle = rank * base + min(rank, remainder) + 1
    last_particle = first_particle + base - 1 + (rank < remainder)
    return first_particle:last_particle
end
