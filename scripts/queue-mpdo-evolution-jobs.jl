
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
    nargs = '+'
    required = true
    "-T"
    arg_type = Int
    nargs = '+'
    required = true
    "-e"
    arg_type = Float64
    nargs = '+'
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
    nargs = '+'
    required = true
    "--user", "-u"
    default = "rdimov"
    "--memoryGB", "-u"
    arg_type = Int
    default = 256
end
args = parse_args(ARGS, s)

Ls = args["L"]
Ts = args["T"]
εs = args["e"]
χs = args["chi"] # outer bond dimension
κs = args["kappa"] # inner bond dimension
rs = args["reps"] # replicas
user = args["user"]
memoryGB = args["memoryGB"]

script_path = joinpath(@__DIR__, "run-time-evolution.jl")

for (L, T, ε, χ, κ, r) in Iterators.product(Ls, Ts, εs, χs, κs, rs)
    scriptargs = [L, T, ε, χ, κ, r, user]
    params = SbatchParams(
        jobname="mpdo-evol",
        memG=memoryGB,
        user=user,
        requeue=true,
        time="12:00:00"
    )
    submitjob(SCRIPTPATH, scriptargs, params)
end
