## IMPORTS ##
using Pkg
Pkg.activate(joinpath(@__DIR__, "../"))
include("../src/time-evolution.jl")
include("../src/results.jl")

using ArgParse

ITensors.set_warn_order(50)

# Commandline arguments
s = ArgParseSettings()
@add_arg_table! s begin
    "L"
    arg_type = Int
    required = true
    "T"
    arg_type = Int
    required = true
    "ε"
    arg_type = Float64
    required = true
    "χ"
    arg_type = Int
    required = true
    "κ"
    arg_type = Int
    required = true
    "replica"
    arg_type = Int
    required = true
    "user"
    required = true
    "type"
    default = "MPDO"
    "--local"
    action = :store_true
    "--inc"
    arg_type = Int
    default = 1
    "--jobid", "-j"
    required = false
end
args = parse_args(ARGS, s)

# Build path at which to save large data files
savedir = args["local"] ? joinpath(@__DIR__, "..", "data") : joinpath("/scratch", "users", args["user"], "noisyRQCs")

# Run evolution
L, T, ε, χ, κ = args["L"], args["T"], args["ε"], args["χ"], args["κ"]
type = typefromstr(args["type"])
tag = args["replica"]
results = @timed evolve_state(L, T, ε, χ, κ, savedir; tag, save_increment=args["inc"], type)

# Save perfomance stats
println("Saving benchmarks...")
benchmarks_filename = "benchmarks-$(paramstring(L, T, ε, χ, κ, type, tag=tag)).csv"
benchmarks_path = joinpath(savedir, benchmarks_filename)
df = load_performance_dataframe(benchmarks_path)
update_performance!(df, L=L, ε=ε, max_outer_dim=χ, max_inner_dim=κ, results=results)
CSV.write(benchmarks_path, df)
println("\nDone!\n")
