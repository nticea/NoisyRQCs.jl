## IMPORTS ##
using Pkg
Pkg.activate(joinpath(@__DIR__, "../../"))
include("../../src/MPDO.jl")
include("../../src/results.jl")

ITensors.set_warn_order(50)
L = 11;
T = 20;
ε = 1e-3;
χ = 32;
κ = 1;
replica = 1;

## PERFORM SHORT RUN TO FORCE COMPILATION ## 
# Initialize the wavefunction to product state (all 0)
# println("Forcing compilation with a small run!")
# ψ0 = initialize_wavefunction(L=5)
# apply_circuit_mpdo(ψ0, 20, ε=ε, maxdim=4,
#     max_inner_dim=κ, benchmark=true, normalize_ρ=true)
## END FORCED COMPILATION ## 

## SAVING INFO ## 
stamp = "results_$(L)L_$(ε)ε_$(χ)max_outer_dim_$(κ)max_inner_dim_$(replica)replica.h5"
datapath = joinpath(@__DIR__, "data_profile")
scratchpath = joinpath("\$SCRATCH", "noisyRQCs")
tensorspath = joinpath("\$SCRATCH", "noisyRQCs")
mkpath(datapath)
mkpath(scratchpath)
savepath = joinpath(datapath, stamp)
benchmarkpath = joinpath(datapath, "performance_L$L.csv")
## END SAVING INFO ##

## RUN THE CODE ## 
# Initialize the wavefunction to product state (all 0)
ψ0 = initialize_wavefunction(L=L)

# Apply the MPDO circuit
all_results = @timed apply_circuit_mpdo(ψ0, T, ε=ε, maxdim=χ,
    max_inner_dim=κ, benchmark=false, normalize_ρ=true, save_path=savepath)
ψ, state_entanglement, operator_entanglement, logneg, MI, trace = all_results.value

## SAVE RESULTS 
# make a results struct and save it 
results = Results(0, L, T, ε, χ, κ, state_entanglement, operator_entanglement, trace, logneg, MI)
save_structs(results, savepath)
println("Successfully saved results to $savepath")

# update performance stats 
df = load_performance_dataframe(benchmarkpath)
update_performance!(df, L=L, ε=ε,
    max_outer_dim=χ, max_inner_dim=κ, results=all_results)
CSV.write(benchmarkpath, df)
## END SAVE RESULTS 


