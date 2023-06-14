## IMPORTS ##
using Pkg
Pkg.activate(joinpath(@__DIR__, "../../"))
include("../../src/MPDO.jl")
include("../../src/circuit.jl")
include("../../src/results.jl")
using Dates

ITensors.set_warn_order(50)

## PARAMETERS ## 
α = nothing #0.9
Ls = [11]
T = 20
max_outer_dims = [30]
max_inner_dims = [50, 58, 64]
normalize_ρ = true
multithread = false

savedir = joinpath(@__DIR__, "data")

function mpdo_circuit_run(; α, Ls, T, max_outer_dim, max_inner_dims, normalize_ρ::Bool, multithread::Bool)
    for L in Ls

        # Initialize the wavefunction to product state (all 0)
        ψ0 = initialize_wavefunction(L=L)
        # calculate ε as a function of L and α
        #ε = α / L
        ε = 1e-3

        for max_inner_dim in max_inner_dims
            GC.gc()

            @show ε, max_inner_dim, max_outer_dim
            # Apply the MPDO circuit
            ψ, state_entanglement, operator_entanglement, logneg, MI, trace = @time apply_circuit_mpdo(ψ0, T, ε=ε, maxdim=max_outer_dim,
                max_inner_dim=max_inner_dim, benchmark=true, normalize_ρ=normalize_ρ, multithread=multithread)

            # make a results struct and save it 
            results = Results(L, T, ε, max_outer_dim, max_inner_dim, state_entanglement, operator_entanglement, trace, logneg, MI)
            timestamp = Dates.format(now(), "yyyy-mm-dd_HH:MM:SS")
            savepath = joinpath(savedir, "results_$(L)Nx_$(α)α_$(max_outer_dim)max_outer_dim_$(max_inner_dim)max_inner_dim" * timestamp * ".h5")
            save_structs(results, savepath)

        end
    end
end

function mpo_circuit_run(; α, Ls, T, max_outer_dims, normalize_ρ::Bool=true)
    for L in Ls

        # Initialize the wavefunction to product state (all 0)
        ψ0 = initialize_wavefunction(L=L)
        # calculate ε as a function of L and α
        # ε = α / L
        ε = 1e-3

        for max_outer_dim in max_outer_dims

            GC.gc()

            @show ε
            # Apply the MPO circuit
            ψ, state_entanglement, operator_entanglement, logneg, MI, trace = @time apply_circuit(ψ0, T, ε=ε, maxdim=max_outer_dim^2,
                benchmark=true, normalize_ρ=normalize_ρ)

            # print results
            flush(stdout)

            # make a results struct and save it 
            results = Results(L, T, ε, maxlinkdim(ψ), 0, state_entanglement, operator_entanglement, trace, logneg, MI)
            timestamp = Dates.format(now(), "yyyy-mm-dd_HH:MM:SS")
            savepath = joinpath(savedir, "results_$(L)Nx_$(α)α_$(max_outer_dim)max_outer_dim_" * timestamp * ".h5")
            save_structs(results, savepath)
        end
    end
end

# actually run the script 
for max_outer_dim in max_outer_dims
    mpdo_circuit_run(α=α, Ls=Ls, T=T, max_outer_dim=max_outer_dim, max_inner_dims=max_inner_dims, normalize_ρ=normalize_ρ, multithread=multithread)
    mpo_circuit_run(α=α, Ls=Ls, T=T, max_outer_dims=max_outer_dims, normalize_ρ=normalize_ρ)
end
