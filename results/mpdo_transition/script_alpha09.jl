## IMPORTS ##
using Pkg
Pkg.activate(joinpath(@__DIR__, "../../"))
include("../../src/MPDO.jl")
include("../../src/results.jl")
using Dates

ITensors.set_warn_order(50)

## PARAMETERS ## 
α = 0.9
Ls = [15, 17, 19]
T = 20
mdim = 16
max_inner_dims = [16]#[16, 18, 20]
normalize_ρ = true
multithread = false

savedir = joinpath(@__DIR__, "data")

function mpdo_circuit_run(; α, Ls, T, maxdim, max_inner_dims, normalize_ρ::Bool, multithread::Bool)
    for L in Ls

        # Initialize the wavefunction to product state (all 0)
        ψ0 = initialize_wavefunction(L=L)
        # calculate ε as a function of L and α
        #ε = α / L
        ε = 0.01

        for max_inner_dim in max_inner_dims

            @show ε, max_inner_dim
            # Apply the MPDO circuit
            ψ, state_entanglement, operator_entanglement, logneg, MI, trace = @time apply_circuit_mpdo(ψ0, T, ε=ε, maxdim=maxdim,
                max_inner_dim=max_inner_dim, benchmark=true, normalize_ρ=normalize_ρ, multithread=multithread)

            # print results
            flush(stdout)

            # make a results struct and save it 
            results = Results(L, T, ε, maxdim, max_inner_dim, state_entanglement, operator_entanglement, trace, logneg, MI)
            timestamp = Dates.format(now(), "yyyy-mm-dd_HH:MM:SS")
            savepath = joinpath(savedir, "results_$(L)Nx_$(α)α_$(max_inner_dim)max_inner_dim" * timestamp * ".h5")
            save_structs(results, savepath)
        end
    end
end

# actually run the script 
for _ in 1:9
    mpdo_circuit_run(α=α, Ls=Ls, T=T, maxdim=mdim, max_inner_dims=max_inner_dims, normalize_ρ=normalize_ρ, multithread=multithread)
end
