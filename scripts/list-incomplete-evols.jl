using Pkg
Pkg.activate(joinpath(@__DIR__, "../"))

using CSV
using DataFrames
using Glob
using ArgParse

include("../src/file-parsing.jl")

s = ArgParseSettings()
@add_arg_table! s begin
    "statedirs"
    nargs = '+'
    required = true
end
args = parse_args(ARGS, s)
statedirs = args["statedirs"]


incompletedirs = []
for statedir in statedirs
    # Find T
    statedirname = splitpath(statedir)[end]
    stateparams = build_state_params(statedirname)
    T = stateparams["T"]

    # Find max t
    statefiles = readdir(statedir)
    if ~isempty(statefiles)
        maxt = maximum(get_t.(basename.(statefiles)))
        if maxt < T
            push!(incompletedirs, statedir)
        end
    end
end

if isempty(incompletedirs)
    println("All state evolutions completed!\n")
else
    println("Incomplete state evolutions:\n")
    for dir in incompletedirs
        println(dir)
    end
end
