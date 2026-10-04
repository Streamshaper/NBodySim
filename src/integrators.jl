abstract type AbstractIntegrator end

struct SemiImplicitEuler <: AbstractIntegrator end

integrator_name(::AbstractIntegrator) = error("Integrator name is not implemented")
integrator_name(::SemiImplicitEuler) = "semi_implicit_euler"

function integrate_particle!(::AbstractIntegrator, args...)
    error("Particle update is not implemented for this integrator")
end

@inline function integrate_particle!(::SemiImplicitEuler,
                                     next_pos::Matrix{Float64},
                                     next_vel::Matrix{Float64},
                                     pos::Matrix{Float64},
                                     vel::Matrix{Float64},
                                     index::Int,
                                     acceleration::NTuple{3, Float64},
                                     timestep::Float64)
    @inbounds begin
        vx = vel[1, index] + acceleration[1] * timestep
        vy = vel[2, index] + acceleration[2] * timestep
        vz = vel[3, index] + acceleration[3] * timestep

        next_vel[1, index] = vx
        next_vel[2, index] = vy
        next_vel[3, index] = vz
        next_pos[1, index] = pos[1, index] + vx * timestep
        next_pos[2, index] = pos[2, index] + vy * timestep
        next_pos[3, index] = pos[3, index] + vz * timestep
    end
    return nothing
end
