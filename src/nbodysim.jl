if !@isdefined __NBodySimCommonLoaded
    include("simulation_common.jl")
    global __NBodySimCommonLoaded = true
end

include("direct.jl")
include("barneshut.jl")
include("fmm.jl")
include("cli_args.jl")

function run_selected_model(model_name::Union{Nothing,AbstractString}=nothing,
                           profile_path::AbstractString="profiles/default.toml",
                           steps_override::Union{Nothing,Int}=nothing)
    profile_file = isabspath(profile_path) ? profile_path : joinpath(@__DIR__, "..", profile_path)
    profile = load_profile(profile_file)
    solver = selected_solver(model_name, profile.solver)
    model = solver_name(solver)

    if model == "direct"
        return run_direct_simulation(profile_path, steps_override)
    elseif model == "barneshut"
        return run_barneshut_simulation(profile_path, steps_override)
    elseif model == "fmm"
        return run_fmm_simulation(profile_path, steps_override)
    end

    error("Unsupported simulation model '$model_name'")
end

if abspath(PROGRAM_FILE) == @__FILE__
    model, profile_path, steps_override = parse_cli_args(ARGS)
    run_selected_model(model, profile_path, steps_override)
end
