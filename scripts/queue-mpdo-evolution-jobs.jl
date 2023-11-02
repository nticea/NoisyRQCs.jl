
using Pkg
Pkg.activate(joinpath(@__DIR__, "../"))

using ArgParse
include("../src/submitjob.jl")

SCRIPTPATH = joinpath(@__DIR__, "run-time-evolution.jl")

# Commandline arguments
s = ArgParseSettings()
@add_arg_table! s begin
    "-L"
    arg_type = Int
    required = true
    "-T"
    arg_type = Int
    required = true
    "-e"
    arg_type = Float64
    required = true
    "--chi", "-c"
    arg_type = Int
    nargs = '+'
    required = true
    "--kappa", "-k"
    arg_type = Int
    nargs = '+'
    required = true
    "--reps", "-r"
    arg_type = Int
    default = 1
    "--user", "-u"
    default = "rdimov"
end
args = parse_args(ARGS, s)

nreps = args["reps"] # number of replicas
L = args["L"]
T = args["T"]
ε = args["e"]
χs = args["chi"] # outer bond dimension
κs = args["kappa"] # inner bond dimension. Set κ=0 → exact simulation of MPO
user = args["user"]

script_path = joinpath(@__DIR__, "run-time-evolution.jl")

for r in 1:nreps
    for χ in χs
        for κ in κs
            scriptargs = [L, T, ε, χ, κ, r, user]
            params = SbatchParams(
                jobname="mpdo-evol",
                memG=256,
                user=user,
                requeue=true
            )
            submitjob(SCRIPTPATH, scriptargs, params)
        end
    end
end
