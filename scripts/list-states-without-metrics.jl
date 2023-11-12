using Pkg
Pkg.activate(joinpath(@__DIR__, "../"))

using CSV
using DataFrames
using Glob
using ArgParse

s = ArgParseSettings()
@add_arg_table! s begin
    "-c", "--combined"
    required = true
    "statedirs"
    nargs = '+'
    required = true
end
args = parse_args(ARGS, s)
statedirs = args["statedirs"]

include("../src/file-parsing.jl")

df = CSV.File(combpath) |> DataFrame

#  TODO: finish

nometrics = []
for statedir in statedirs
    # Find T
    statedirname = splitpath(statedir)[end]
    stateparams = build_state_params(statedirname)

    statefiles = readdir(statedir)
    for statefile in statefiles
        t = get_t(basename(statefile))
    end
end
