
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using ArgParse
include("../src/submitjob.jl")

SCRIPTPATH = joinpath(@__DIR__, "compute-metrics.jl")

# Commandline arguments
s = ArgParseSettings()
@add_arg_table! s begin
    "statepaths"
    action = :store_arg
    nargs = '+'
    required = true
    "--user", "-u"
    default = "rdimov"
    "--chunk", "-c"
    arg_type = Int
    default = 10
end
args = parse_args(ARGS, s)

# Get the absolute path
statepaths = joinpath.(Ref(pwd()), args["statepaths"])

for statepathschunk in Iterators.partition(statepaths, args["chunk"])
    params = SbatchParams(
        jobname="mpdo-metrics",
        memG=256,
        user=args["user"],
    )
    scriptargs = statepathschunk
    submitjob(SCRIPTPATH, scriptargs, params)
end

println("Queued $(Int(ceil(length(statepaths) / args["chunk"]))) jobs!\n")
