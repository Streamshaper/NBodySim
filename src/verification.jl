using LinearAlgebra
using Dates

# Conserved quantities used to measure drift.
struct VerificationReference
    total_mass::Float64
    center_of_mass::Vector{Float64}
    momentum::Vector{Float64}
    angular_momentum::Vector{Float64}
    energy::Float64
    position_scale::Float64
    velocity_scale::Float64
end

# Total kinetic + potential energy for the verification budget.
function _verification_energy(pos::Matrix{Float64}, vel::Matrix{Float64}, masses::Vector{Float64},
                              interaction_strength::Float64, smoothing::Float64,
                              max_particles::Int)
    n_particles = size(pos, 2)
    n_particles <= max_particles || return NaN

    kinetic = 0.0
    potential = 0.0
    for i in 1:n_particles
        kinetic += 0.5 * masses[i] * sum(abs2, @view vel[:, i])
        for j in 1:(i - 1)
            displacement = pos[:, i] - pos[:, j]
            potential -= interaction_strength * masses[i] * masses[j] /
                         sqrt(sum(abs2, displacement) + smoothing^2)
        end
    end
    return kinetic + potential
end

# Baseline invariants for COM, momentum, angular momentum, and energy.
function verification_reference(pos::Matrix{Float64}, vel::Matrix{Float64}, masses::Vector{Float64},
                               interaction_strength::Float64, smoothing::Float64,
                               max_energy_particles::Int)
    total_mass = sum(masses)
    center_of_mass = vec(pos * masses) ./ total_mass
    momentum = vec(vel * masses)
    angular_momentum = vec(sum(cross(pos[:, i], masses[i] .* vel[:, i]) for i in axes(pos, 2)))
    position_scale = max(maximum(norm.(eachcol(pos))), 1.0)
    velocity_scale = max(maximum(norm.(eachcol(vel))), 1.0)
    energy = _verification_energy(pos, vel, masses, interaction_strength, smoothing,
                                  max_energy_particles)
    return VerificationReference(total_mass, center_of_mass, momentum, angular_momentum,
                                 energy, position_scale, velocity_scale)
end

# Drift from the reference conserved quantities.
function verification_metrics(pos::Matrix{Float64}, vel::Matrix{Float64}, masses::Vector{Float64},
                              interaction_strength::Float64, smoothing::Float64,
                              reference::VerificationReference, max_energy_particles::Int)
    center_of_mass = vec(pos * masses) ./ reference.total_mass
    momentum = vec(vel * masses)
    angular_momentum = vec(sum(cross(pos[:, i], masses[i] .* vel[:, i]) for i in axes(pos, 2)))
    energy = _verification_energy(pos, vel, masses, interaction_strength, smoothing,
                                  max_energy_particles)

    return (
        energy = energy,
        relative_energy_change = isfinite(energy) && isfinite(reference.energy) ?
            (energy - reference.energy) / max(abs(reference.energy), eps(Float64)) : NaN,
        center_of_mass_drift = norm(center_of_mass - reference.center_of_mass) / reference.position_scale,
        relative_momentum_change = norm(momentum - reference.momentum) /
                                   max(norm(reference.momentum), reference.total_mass * reference.velocity_scale * eps(Float64)),
        relative_angular_momentum_change = norm(angular_momentum - reference.angular_momentum) /
                                           max(norm(reference.angular_momentum), reference.total_mass * reference.position_scale * reference.velocity_scale * eps(Float64))
    )
end

function write_verification_header(io)
    println(io, "step,time,energy,relative_energy_change,center_of_mass_drift,relative_momentum_change,relative_angular_momentum_change")
end

function write_verification_row(io, step::Int, time::Float64, metrics)
    println(io, join((step, time, metrics.energy, metrics.relative_energy_change,
                      metrics.center_of_mass_drift, metrics.relative_momentum_change,
                      metrics.relative_angular_momentum_change), ','))
end

function verification_log_path()
    date_directory = Dates.format(Dates.today(), dateformat"yyyy-mm-dd")
    job_id = get(ENV, "SLURM_JOB_ID", "local")
    filename = "verification_$(job_id).csv"
    return joinpath("logs", date_directory, filename)
end
