## IMPORTS ##
using Pkg
Pkg.activate(joinpath(@__DIR__, "../../"))
include("../../src/MPDO.jl")
include("../../src/results.jl")
using Dates

ITensors.set_warn_order(50)

## PARAMETERS ## 
L = 9
T = 20
εs = [1e-2]
mdim = 16
max_inner_dims = [16]
normalize_ρ = true

savedir = joinpath(@__DIR__, "data")

function mpdo_circuit_run(; L, T, maxdim, εs, max_inner_dims, normalize_ρ::Bool=true)
    # Initialize the wavefunction to product state (all 0)
    ψ0 = initialize_wavefunction(L=L)

    for ε in εs
        for max_inner_dim in max_inner_dims
            @show ε, max_inner_dim
            # Apply the MPDO circuit
            ψ, state_entanglement, operator_entanglement, logneg, MI, trace = @time apply_circuit_mpdo(ψ0, T, ε=ε, maxdim=maxdim,
                max_inner_dim=max_inner_dim, benchmark=true, normalize_ρ=normalize_ρ)

            # make a results struct and save it 
            results = Results(L, T, ε, maxdim, max_inner_dim, state_entanglement, operator_entanglement, trace, logneg, MI)
            timestamp = Dates.format(now(), "yyyy-mm-dd_HH:MM:SS")
            savepath = joinpath(savedir, "results_$(L)Nx_$(ε)ε_$(max_inner_dim)max_inner_dim" * timestamp * ".h5")
            save_structs(results, savepath)
        end
    end
end

# actually run the script 
for _ in 1:10
    mpdo_circuit_run(L=L, T=T, maxdim=mdim, εs=εs, max_inner_dims=[16], normalize_ρ=normalize_ρ)
end

for _ in 1:10
    mpdo_circuit_run(L=L, T=T, maxdim=mdim, εs=εs, max_inner_dims=[40], normalize_ρ=normalize_ρ)
end