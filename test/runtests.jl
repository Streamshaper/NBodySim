using Test

include(joinpath(@__DIR__, "..", "src", "profile.jl"))
include(joinpath(@__DIR__, "..", "src", "verification.jl"))
include(joinpath(@__DIR__, "..", "src", "logging.jl"))

@testset "Semi-implicit Euler integrator" begin
    pos = [1.0; 2.0; 3.0;;]
    vel = [4.0; 5.0; 6.0;;]
    next_pos = zeros(3, 1)
    next_vel = zeros(3, 1)
    integrate_particle!(SemiImplicitEuler(), next_pos, next_vel, pos, vel, 1,
                        (0.5, -1.0, 2.0), 0.25)
    @test next_vel[:, 1] ≈ [4.125, 4.75, 6.5]
    @test next_pos[:, 1] ≈ [2.03125, 3.1875, 4.625]
    @test pos[:, 1] == [1.0, 2.0, 3.0]
    @test vel[:, 1] == [4.0, 5.0, 6.0]
    @test integrator_name(SemiImplicitEuler()) == "semi_implicit_euler"
end

@testset "Interaction kernels" begin
    plummer = PlummerGravity(2.0, 0.5)
    @test kernel_acceleration(plummer, 3.0, 1.0, 0.0, 0.0) ==
          (2.0 * 3.0 / 1.25^1.5, 0.0, 0.0)
    @test pair_potential(plummer, 2.0, 3.0, 1.0, 0.0, 0.0) ≈
          -12.0 / sqrt(1.25)

    yukawa = YukawaGravity(2.0, 0.0, 4.0)
    @test collect(kernel_acceleration(yukawa, 3.0, 2.0, 0.0, 0.0)) ≈
          [12.0 * exp(-0.5) * (1 / 8 + 1 / 16), 0.0, 0.0]
    @test pair_potential(yukawa, 2.0, 3.0, 2.0, 0.0, 0.0) ≈
          -6.0 * exp(-0.5)
    @test circular_orbital_speed(plummer, 4.0, 2.0) == 2.0
    @test circular_orbital_speed(yukawa, 4.0, 2.0) > 0

    positions = [0.0 1.0; 0.0 0.0; 0.0 0.0]
    velocities = zeros(3, 2)
    masses = [2.0, 3.0]
    reference = verification_reference(positions, velocities, masses, plummer, 2)
    @test reference.energy ≈ -12.0 / sqrt(1.25)
    metrics = verification_metrics(positions, velocities, masses, plummer, reference, 2)
    @test metrics.relative_energy_change == 0.0

    @test supports_kernel(Val(:direct), yukawa)
    @test !supports_kernel(Val(:barneshut), yukawa)
    @test_throws ErrorException require_kernel_support(:barneshut, yukawa)
end

@testset "Versioned stopwatch logging" begin
    mktempdir() do temp_dir
        cd(temp_dir) do
            mkpath("logs")
            write(joinpath("logs", "stopwatch.csv"), "legacy header\nlegacy row\n")

            withenv("SLURM_CPUS_PER_TASK" => "4", "SLURM_JOB_NUM_NODES" => "2") do
                write_log(DirectSolver(), PlummerGravity(2.0, 0.5), SemiImplicitEuler(),
                          10, 20, 1.25, 0.0)
            end

            legacy_path = joinpath("logs", "stopwatch.csv")
            @test read(legacy_path, String) == "legacy header\nlegacy row\n"

            versioned_path = joinpath("logs", "stopwatch_v2.csv")
            lines = readlines(versioned_path)
            @test lines[1] == STOPWATCH_V2_HEADER
            @test lines[2] ==
                  "direct,plummer_gravity,semi_implicit_euler,2,4,10,20,1.25,0.00"

            withenv("SLURM_CPUS_PER_TASK" => "1", "SLURM_JOB_NUM_NODES" => "1") do
                write_log(FMMSolver(), PlummerGravity(2.0, 0.5), SemiImplicitEuler(),
                          10, 20, 2.0, 0.5)
            end
            @test length(readlines(versioned_path)) == 3

            write(versioned_path, "unexpected header\n")
            @test_throws ErrorException write_log(
                DirectSolver(), PlummerGravity(2.0, 0.5), SemiImplicitEuler(),
                10, 20, 1.25, 0.0)
            @test read(versioned_path, String) == "unexpected header\n"
        end
    end
end

@testset "Kernel profile loading" begin
    default_profile = load_profile(joinpath(@__DIR__, "..", "profiles", "default.toml"))
    @test default_profile.kernel ==
          PlummerGravity(default_profile.interaction_strength, default_profile.smoothing)

    base_profile = Dict(
        "simulation" => Dict("interaction_strength" => 3.0, "smoothing" => 0.25),
        "particles" => Dict(
            "positions" => [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            "velocities" => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            "masses" => [1.0, 2.0],
        ),
    )

    mktemp() do path, io
        TOML.print(io, base_profile)
        close(io)
        profile = load_profile(path)
        @test profile.kernel == PlummerGravity(3.0, 0.25)
        @test profile.integrator == SemiImplicitEuler()
        @test profile.solver == DirectSolver()
        @test profile.interaction_strength == 3.0
        @test profile.smoothing == 0.25
    end

    for (solver_type, expected_solver) in (
        ("barneshut", BarnesHutSolver()),
        ("fmm", FMMSolver()),
    )
        solver_profile = deepcopy(base_profile)
        solver_profile["solver"] = Dict("type" => solver_type)
        mktemp() do path, io
            TOML.print(io, solver_profile)
            close(io)
            profile = load_profile(path)
            @test profile.solver == expected_solver
        end
    end

    yukawa_profile = deepcopy(base_profile)
    yukawa_profile["kernel"] = Dict(
        "type" => "yukawa_gravity",
        "strength" => 5.0,
        "smoothing" => 0.5,
        "screening_length" => 12.0,
    )
    yukawa_profile["solver"] = Dict("type" => "direct")
    mktemp() do path, io
        TOML.print(io, yukawa_profile)
        close(io)
        profile = load_profile(path)
        @test profile.kernel == YukawaGravity(5.0, 0.5, 12.0)
        @test profile.integrator == SemiImplicitEuler()
        @test profile.solver == DirectSolver()
    end

    unsupported_profile = deepcopy(base_profile)
    unsupported_profile["integrator"] = Dict("type" => "unsupported")
    mktemp() do path, io
        TOML.print(io, unsupported_profile)
        close(io)
        @test_throws ErrorException load_profile(path)
    end

    incompatible_profile = deepcopy(yukawa_profile)
    incompatible_profile["solver"] = Dict("type" => "barneshut")
    mktemp() do path, io
        TOML.print(io, incompatible_profile)
        close(io)
        @test_throws ErrorException load_profile(path)
    end
end

@testset "Typed solver selection" begin
    @test solver_name(parse_solver("direct")) == "direct"
    @test parse_solver("bh") == BarnesHutSolver()
    @test parse_solver("f") == FMMSolver()
    @test selected_solver(nothing, BarnesHutSolver()) == BarnesHutSolver()
    @test selected_solver("fmm", DirectSolver()) == FMMSolver()
    @test_throws ErrorException parse_solver("unknown")
    @test supports_integrator(BarnesHutSolver(), SemiImplicitEuler())
end
