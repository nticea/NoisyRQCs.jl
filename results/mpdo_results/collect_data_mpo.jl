## IMPORTS ##
using Pkg
Pkg.activate(joinpath(@__DIR__, "../../"))
include("../../src/circuit.jl")
include("../../src/results.jl")

ITensors.set_warn_order(50)

## PARAMETERS ## 
args = parse.(Float64, ARGS)
@assert length(args) == 6
L, T, ε, χ, κ, replica = [args[n] for n in 1:length(args)]
L, T, χ, κ, replica = Int(L), Int(T), Int(χ), Int(κ), Int(replica)

stamp = "results_$(L)L_$(ε)ε_$(χ)max_outer_dim_$(0)max_inner_dim_$(replica)replica.h5"

datapath = joinpath(@__DIR__, "data")
scratchpath = joinpath("\$SCRATCH", "noisyRQCs")
mkpath(datapath)
mkpath(scratchpath)

savepath = joinpath(datapath, stamp)
benchmarkpath = joinpath(datapath, "performance_L$L.csv")
checkptpath = joinpath(scratchpath, stamp)

# Initialize the wavefunction to product state (all 0)
ψ0 = initialize_wavefunction(L=L)

# Apply the MPO circuit
all_results = @timed apply_circuit(ψ0, T, ε=ε, maxdim=χ^2,
    benchmark=true, normalize_ρ=true, checkpoint_path=checkptpath, save_path=savepath)
ψ, state_entanglement, operator_entanglement, logneg, MI, trace = all_results.value

# make a results struct and save it 
results = Results(0, L, T, ε, χ^2, 0, state_entanglement, operator_entanglement, trace, logneg, MI)
save_structs(results, savepath)

# update performance stats 
df = load_performance_dataframe(benchmarkpath)
update_performance!(df, L=L, ε=ε,
    max_outer_dim=χ, max_inner_dim=0, results=all_results)
CSV.write(benchmarkpath, df)
