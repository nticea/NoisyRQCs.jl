using Pkg
Pkg.activate(joinpath(@__DIR__, "../"))

using CSV
using DataFrames
using Glob
using ArgParse

s = ArgParseSettings()
@add_arg_table! s begin
    "statedirs"
    nargs = '+'
    required = true
end
args = parse_args(ARGS, s)
statedirs = args["statedirs"]

function build_state_params(statedirname)
    # match any digits and a single decimal point
    pattern = r"(\d+(\.\d+)?)(L|T|noise|outer|inner)"
    matches = eachmatch(pattern, statedirname)
    return Dict(m[3] => parse(contains(m[1], ".") ? Float64 : Int, m[1]) for m in matches)
end

function get_t(statefilename)
    pattern = r"state_t(\d+)"
    m = match(pattern, statefilename)
    return parse(Int, m[1])
end

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
