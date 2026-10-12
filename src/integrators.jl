abstract type AbstractIntegrator end

struct SemiImplicitEuler <: AbstractIntegrator end

integrator_name(::AbstractIntegrator) = error("Integrator name is not implemented")
integrator_name(::SemiImplicitEuler) = "semi_implicit_euler"

function integrate_particle!(::AbstractIntegrator, args...)
    error("Particle update is not implemented for this integrator")
end

@inline function integrate_particle!(integrator::SemiImplicitEuler,
                                     next_pos::Matrix{Float64},
                                     next_vel::Matrix{Float64},
                                     pos::Matrix{Float64},
                                     vel::Matrix{Float64},
                                     index::Int,
                                     acceleration::NTuple{3, Float64},
                                     timestep::Float64)
    return integrate_particle!(integrator, next_pos, next_vel, index, pos, vel,
                                index, acceleration, timestep)
end

@inline function integrate_particle!(::SemiImplicitEuler,
                                     next_pos::Matrix{Float64},
                                     next_vel::Matrix{Float64},
                                     output_index::Int,
                                     pos::Matrix{Float64},
                                     vel::Matrix{Float64},
                                     input_index::Int,
                                     acceleration::NTuple{3, Float64},
                                     timestep::Float64)
    @inbounds begin
        vx = vel[1, input_index] + acceleration[1] * timestep
        vy = vel[2, input_index] + acceleration[2] * timestep
        vz = vel[3, input_index] + acceleration[3] * timestep

        next_vel[1, output_index] = vx
        next_vel[2, output_index] = vy
        next_vel[3, output_index] = vz
        next_pos[1, output_index] = pos[1, input_index] + vx * timestep
        next_pos[2, output_index] = pos[2, input_index] + vy * timestep
        next_pos[3, output_index] = pos[3, input_index] + vz * timestep
    end
    return next_pos[3, output_index]
end
