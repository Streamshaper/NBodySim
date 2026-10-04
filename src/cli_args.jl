resolve_model(model_name::AbstractString) = solver_name(parse_solver(model_name))

function parse_cli_args(args::Vector{String})
    if isempty(args)
        println("Usage: julia --project=. src/nbodysim.jl <profile> [solver] [steps]")
        return nothing, "profiles/default.toml", nothing
    end

    length(args) <= 3 || error("Too many arguments. Expected <profile> [solver] [steps]")

    first = args[1]
    if is_solver_name(first)
        model = resolve_model(first)
        profile = length(args) >= 2 ? args[2] : "profiles/default.toml"
        steps = length(args) >= 3 ? parse(Int, args[3]) : nothing
        return model, profile, steps
    end

    profile = first
    if length(args) == 1
        return nothing, profile, nothing
    elseif is_solver_name(args[2])
        model = resolve_model(args[2])
        steps = length(args) >= 3 ? parse(Int, args[3]) : nothing
        return model, profile, steps
    elseif length(args) == 2
        return nothing, profile, parse(Int, args[2])
    end

    error("Expected a solver override after the profile when providing three arguments")
end
