## IMPORTS ##
using Pkg
Pkg.activate(joinpath(@__DIR__, "../"))
include("../src/MPDO.jl")
include("../src/time-evolution.jl")

using ArgParse
using DataFrames
using CSV

ITensors.set_warn_order(50)

# Commandline arguments
s = ArgParseSettings()
@add_arg_table! s begin
    "statepath"
    required = true
end
args = parse_args(ARGS, s)
statepath = args["statepath"]
println("state path: ", statepath)

# Build output path
println("Building output path...")
statedir = splitpath(dirname(statepath))[end]
dirpath = joinpath(@__DIR__, "..", "data", "$(statedir)-metrics")
mkpath(dirpath)
base, ext = split(basename(statepath), ".")
metricsfilename = "$(base).csv"
metricspath = joinpath(dirpath, metricsfilename)

# Load state
println("Loading state...")
state = load_state(statepath)

# Compute metrics
println("Computing metrics...")
trace, svn, lns, mis = compute_metrics(state)

# Write metrics to CSV
println("Building dataframe...")
df = DataFrame(trace=[trace], svn=[svn], lns=[lns], mis=[mis])
println("Saving metrics to $(metricspath)...")
CSV.write(metricspath, df)

println("\nDone!\n")
