
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using ArgParse
include("../src/submitjob.jl")

SCRIPTPATH = joinpath(@__DIR__, "compute_mpdo_metrics.jl")

# Commandline arguments
s = ArgParseSettings()
@add_arg_table! s begin
    "statepaths"
    action = :store_arg
    nargs = '+'
    required = true
    "-u", "--user"
    default = "rdimov"
end
args = parse_args(ARGS, s)

# Get the absolute path
statepaths = joinpath.(Ref(pwd()), args["statepaths"])

for statepath in statepaths
    statedir = splitpath(dirname(statepath))[end]
    statename = split(basename(statepath), ".")[1]
    params = SbatchParams(
        jobname="$statedir-$statename-metrics",
        memG=256,
        user="rdimov",
    )
    scriptargs = [statepath]
    submitjob(SCRIPTPATH, scriptargs, params)
end

println("Queued $(length(statepaths)) jobs!\n")
