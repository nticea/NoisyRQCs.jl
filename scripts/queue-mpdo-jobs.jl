## IMPORTS ##
using Pkg
Pkg.activate(joinpath(@__DIR__, "../"))
include("../src/submit_job.jl")

# TODO all combinations
nreps = 5 # number of replicas
L = 5
T = 20
ε = 1e-3
χs = [512] # outer bond dimension
κs = [1, 2, 4, 8] # inner bond dimension. Set κ=0 → exact simulation of MPO
user = ARGS[1]

script_path = joinpath(@__DIR__, "run-time-evolution.jl")

for r in 1:nreps
    for χ in χs
        for κ in κs
            params = RunParams(L, T, ε, χ, κ, r, user)
            submit_job(params, script_path, @__DIR__, "mpdo_sweep")
        end
    end
end
