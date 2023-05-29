## IMPORTS ##
using Pkg
Pkg.activate(joinpath(@__DIR__, "../../"))
include("../../src/circuit.jl")
include("../../src/results.jl")
using Dates

ITensors.set_warn_order(50)

## PARAMETERS ## 
α = 0.9
Ls = [15]
T = 20
mdim = nothing
normalize_ρ = true

savedir = joinpath(@__DIR__, "data")

function mpo_circuit_run(; α, Ls, T, maxdim, normalize_ρ::Bool=true)
    for L in Ls

        # Initialize the wavefunction to product state (all 0)
        ψ0 = initialize_wavefunction(L=L)
        # calculate ε as a function of L and α
        ε = α / L

        @show ε
        # Apply the MPO circuit
        ψ, state_entanglement, operator_entanglement, logneg, MI, trace = @time apply_circuit(ψ0, T, ε=ε, maxdim=maxdim,
            benchmark=true, normalize_ρ=normalize_ρ)

        # print results
        flush(stdout)

        # make a results struct and save it 
        results = Results(L, T, ε, maxlinkdim(ψ), 0, state_entanglement, operator_entanglement, trace, logneg, MI)
        timestamp = Dates.format(now(), "yyyy-mm-dd_HH:MM:SS")
        savepath = joinpath(savedir, "results_$(L)Nx_$(α)α_" * timestamp * ".h5")
        save_structs(results, savepath)
    end
end

# actually run the script 
mpo_circuit_run(α=α, Ls=Ls, T=T, maxdim=mdim, normalize_ρ=normalize_ρ)
