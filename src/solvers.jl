abstract type AbstractSolver end

struct DirectSolver <: AbstractSolver end
struct BarnesHutSolver <: AbstractSolver end
struct FMMSolver <: AbstractSolver end

solver_name(::DirectSolver) = "direct"
solver_name(::BarnesHutSolver) = "barneshut"
solver_name(::FMMSolver) = "fmm"

solver_kind(::DirectSolver) = :direct
solver_kind(::BarnesHutSolver) = :barneshut
solver_kind(::FMMSolver) = :fmm

selected_solver(model_override::Union{Nothing,AbstractString}, configured::AbstractSolver) =
    model_override === nothing ? configured : parse_solver(model_override)

supports_integrator(::AbstractSolver, ::AbstractIntegrator) = false
supports_integrator(::DirectSolver, ::SemiImplicitEuler) = true
supports_integrator(::BarnesHutSolver, ::SemiImplicitEuler) = true
supports_integrator(::FMMSolver, ::SemiImplicitEuler) = true

function require_solver_support(solver::AbstractSolver, kernel::AbstractKernel,
                                integrator::AbstractIntegrator)
    require_kernel_support(solver_kind(solver), kernel)
    supports_integrator(solver, integrator) ||
        error("Solver '$(solver_name(solver))' does not support integrator " *
              "'$(integrator_name(integrator))'")
    return nothing
end

function parse_solver(name::AbstractString)
    normalized = lowercase(name)
    normalized in ("direct", "d") && return DirectSolver()
    normalized in ("barneshut", "bh") && return BarnesHutSolver()
    normalized in ("fmm", "f") && return FMMSolver()
    error("Unknown solver '$name'. Expected one of: direct, barneshut, fmm")
end
