## IMPORTS ##
using Pkg
Pkg.activate(joinpath(@__DIR__, "../"))
include("../src/mpdo.jl")
include("../src/time-evolution.jl")
include("../src/file-parsing.jl")
include("../src/entropy-metrics.jl")

using ArgParse
using DataFrames
using CSV

ITensors.set_warn_order(50)

# Commandline arguments
s = ArgParseSettings()
@add_arg_table! s begin
    "statepaths"
    required = true
    nargs = '+'
    "--jobid", "-j"
    required = false
    default = ""
end
args = parse_args(ARGS, s)
statepaths, jobid = args["statepaths"], args["jobid"]
println("State paths: ", statepaths)
println("Computing metrics for $(length(statepaths)) states")

for statepath in statepaths
    # Build output path
    println("Building output path...")
    statedir = splitpath(dirname(statepath))[end]
    dirpath = joinpath(@__DIR__, "..", "data", "$(statedir)-metrics")
    mkpath(dirpath)
    base, ext = split(basename(statepath), ".")
    metricsfilename = "$(base)-$(jobid).csv"
    metricspath = joinpath(dirpath, metricsfilename)

    # Load state
    println("Loading state...")
    statetype = typefromstr(statetypestr(statedir))
    state = load_state(statepath, statetype)

    # Compute metrics
    println("Computing metrics...")
    trace, svn, lns, mis = compute_metrics(state)

    # Write metrics to CSV
    println("Building dataframe...")
    df = DataFrame(trace=[trace], svn=[svn], lns=[lns], mis=[mis])
    println("Saving metrics to $(metricspath)...")
    CSV.write(metricspath, df)
    println()
end

println("\nDone!\n")
