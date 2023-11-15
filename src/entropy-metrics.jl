using ITensors

function partial_transpose(ρ::ITensor, sites::Vector{<:Index})
    primedsites = prime(sites)
    return replaceinds(ρ, [sites..., primedsites...], [primedsites..., sites...])
end

function trace_norm(ρ::ITensor)
    sites = filter(i -> hasplev(i, 0) & hastags(i, "Site"), inds(ρ))
    U, S, V = ITensors.svd(ρ, sites, cutoff=0)
    return sum(S)
end

function negativity(ρ, B)
    # compute the partial transpose
    ρT = partial_transpose(ρ, B)

    # take the trace norm
    return (trace_norm(ρT) - 1) / 2
end

"""
EN(ρ_AB) = log₂||ρ_AB^(T_B)||₁
"""
function logarithmic_negativity(ρ, B)
    # compute the partial transpose
    ρT = partial_transpose(ρ, B)

    # take the trace norm
    trnorm = trace_norm(ρT)

    # take the logarithm
    return log2(trnorm)
end

function compute_metrics(state)
    # TODO: is there a better way to handle MPS?
    if typeof(state) == MPS
        state = density(state)
    end

    # trace
    trace = real(tr(state))

    # state entanglement
    mid = length(state) ÷ 2
    mid_svn = von_neumann_entropy(state, mid)

    lns = Float64[]
    mis = Float64[]
    i = 2 # start from the second site to avoid edge effects
    i_idx = siteind(state, i)
    for j in (i+1):length(state)
        j_idx = siteind(state, j)
        # compute two-site reduced density
        ρij = twosite_reduced_density(state, i, j)

        # logarithmic negativity
        push!(lns, logarithmic_negativity(ρij, [i_idx]))

        # mutual information
        ρi = ρij * δ(i_idx, i_idx')
        ρj = ρij * δ(j_idx, j_idx')
        push!(mis, mutual_information(ρi, ρj, ρij))

    end
    return trace, mid_svn, lns, mis
end
