abstract type AbstractKernel end

struct PlummerGravity <: AbstractKernel
    interaction_strength::Float64
    smoothing::Float64
end

struct YukawaGravity <: AbstractKernel
    interaction_strength::Float64
    smoothing::Float64
    screening_length::Float64
end

struct Coulomb <: AbstractKernel
    interaction_strength::Float64
    smoothing::Float64
end

struct ParticleProperties
    mass::Float64
    charge::Float64
end

kernel_name(kernel::AbstractKernel) = string(nameof(typeof(kernel)))
kernel_name(::PlummerGravity) = "plummer_gravity"
kernel_name(::YukawaGravity) = "yukawa_gravity"
kernel_name(::Coulomb) = "coulomb"

supports_kernel(::Symbol, ::AbstractKernel) = false
supports_kernel(::Val, ::AbstractKernel) = false
supports_kernel(::Val{:direct}, ::AbstractKernel) = true
supports_kernel(::Val{:barneshut}, ::PlummerGravity) = true
supports_kernel(::Val{:barneshut}, ::Coulomb) = true
supports_kernel(::Val{:fmm}, ::PlummerGravity) = true
supports_kernel(::Val{:fmm}, ::Coulomb) = false
supports_kernel(::Val{:barneshut}, ::YukawaGravity) = false
supports_kernel(::Val{:fmm}, ::YukawaGravity) = false

function require_kernel_support(solver::Symbol, kernel::AbstractKernel)
    supports_kernel(Val(solver), kernel) ||
        error("Solver '$solver' does not support kernel '$(kernel_name(kernel))'")
    return nothing
end

function kernel_acceleration(kernel::PlummerGravity, target::ParticleProperties,
                             source::ParticleProperties,
                             dx::Float64, dy::Float64, dz::Float64)::NTuple{3, Float64}
    distance_squared = dx^2 + dy^2 + dz^2 + kernel.smoothing^2
    scale = kernel.interaction_strength * source.mass / distance_squared^1.5
    return (scale * dx, scale * dy, scale * dz)
end

function kernel_acceleration(kernel::YukawaGravity, target::ParticleProperties,
                             source::ParticleProperties,
                             dx::Float64, dy::Float64, dz::Float64)::NTuple{3, Float64}
    distance_squared = dx^2 + dy^2 + dz^2 + kernel.smoothing^2
    distance = sqrt(distance_squared)
    scale = kernel.interaction_strength * source.mass * exp(-distance / kernel.screening_length) *
            (1 / distance_squared^1.5 + 1 / (kernel.screening_length * distance_squared))
    return (scale * dx, scale * dy, scale * dz)
end

function kernel_acceleration(kernel::Coulomb, target::ParticleProperties,
                             source::ParticleProperties,
                             dx::Float64, dy::Float64, dz::Float64)::NTuple{3, Float64}
    distance_squared = dx^2 + dy^2 + dz^2 + kernel.smoothing^2
    scale = -kernel.interaction_strength * target.charge * source.charge /
            (target.mass * distance_squared^1.5)
    return (scale * dx, scale * dy, scale * dz)
end

function pair_potential(kernel::PlummerGravity, particle_a::ParticleProperties,
                        particle_b::ParticleProperties,
                        dx::Float64, dy::Float64, dz::Float64)::Float64
    distance = sqrt(dx^2 + dy^2 + dz^2 + kernel.smoothing^2)
    return -kernel.interaction_strength * particle_a.mass * particle_b.mass / distance
end

function pair_potential(kernel::YukawaGravity, particle_a::ParticleProperties,
                        particle_b::ParticleProperties,
                        dx::Float64, dy::Float64, dz::Float64)::Float64
    distance = sqrt(dx^2 + dy^2 + dz^2 + kernel.smoothing^2)
    return -kernel.interaction_strength * particle_a.mass * particle_b.mass *
           exp(-distance / kernel.screening_length) / distance
end

function pair_potential(kernel::Coulomb, particle_a::ParticleProperties,
                        particle_b::ParticleProperties,
                        dx::Float64, dy::Float64, dz::Float64)::Float64
    distance = sqrt(dx^2 + dy^2 + dz^2 + kernel.smoothing^2)
    return kernel.interaction_strength * particle_a.charge * particle_b.charge / distance
end

circular_orbital_speed(kernel::PlummerGravity, central_mass::Float64,
                       radius::Float64) =
    sqrt(kernel.interaction_strength * central_mass / radius)

function circular_orbital_speed(kernel::YukawaGravity, central_mass::Float64,
                                radius::Float64)
    radial_acceleration = kernel_acceleration(
        kernel, ParticleProperties(1.0, 0.0), ParticleProperties(central_mass, 0.0),
        radius, 0.0, 0.0)[1]
    return sqrt(radius * radial_acceleration)
end

function circular_orbital_speed(::AbstractKernel, ::Float64, ::Float64)
    error("This kernel has no generated orbital initializer; provide a fixed particle state")
end
