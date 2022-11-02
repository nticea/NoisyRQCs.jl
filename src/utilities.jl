using ITensors

function initialize_wavefunction(;L::Int)
    @assert isodd(L) "L must be odd"
    sites = siteinds("Qubit", L)
    state_arr = ["0" for l=1:L]
    productMPS(sites,state_arr) 
end

function combine_indices(ρ::MPO)
    ρ = copy(ρ)
    orthogonalize!(ρ,1)
    # Combine the primed and unprimed indices at each site to create a super MPS 
    sites = siteinds(ρ)
    for i in 1:length(ρ)
        C = combiner(sites[i]...)
        ρ[i] *= C
    end

    # Put this data into an MPS struct 
    ψ = MPS(ρ.data) 

    return ψ
end

function measure_computational_basis(ρ::MPO; nsamples=10)
    ψ = combine_indices(ρ)

    # Sample multiple times from this super MPS 
    samples = []
    for _ in 1:nsamples
        push!(samples, ITensors.sample!(ψ))
    end
    return cat(samples, dims=2)
end