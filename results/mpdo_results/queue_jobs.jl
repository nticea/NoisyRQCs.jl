## IMPORTS ##
using Pkg
Pkg.activate(joinpath(@__DIR__, "../../"))
include("../../src/submit_job.jl")

nreps = 5 # number of replicas 
L = 13
T = 20
ε = 1e-4
χs = [30, 45, 55, 64]
κs = [0, 1, 2, 4] # 0 is just the MPO, so it is exact

filepath_MPO = joinpath(@__DIR__, "collect_data_mpo.jl")
filepath_MPDO = joinpath(@__DIR__, "collect_data_mpdo.jl")

for r in 1:nreps
    for χ in χs
        for κ in κs
            # make the model parameters
            params = RunParams(L, T, ε, χ, κ, r)

            # pass them into the job dispatch function 
            if κ == 0
                submit_job(params, filepath_MPO, @__DIR__, "mpo_sweep")
            else
                submit_job(params, filepath_MPDO, @__DIR__, "mpdo_sweep")
            end
        end
    end
end