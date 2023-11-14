using LinearAlgebra
using ITensors

function LinearAlgebra.tr(state::MPO, range)
    return tr(ITensor(1.0), state, range)
end

function LinearAlgebra.tr(curr::ITensor, state::MPO, range)
    for i in range
        curr *= tr(state[i], plev=0 => 1)
    end
    return curr
end

function von_neumann_entropy(state::MPO, i)
    orthogonalize!(state, i)
    U, S, V = svd(state[i], [linkind(state, i - 1), siteind(state, i), siteind(state, i)'])
    SvN = 0.0
    for n = 1:ITensors.dim(S, 1)
        p = S[n, n]^2
        if p ≈ 0
            SvN -= 0
        else
            SvN -= p * log2(p)
        end
    end
    return SvN
end

function twosite_reduced_density(state::MPO, site1::Int, site2::Int; tom=false)
    # Ensure left site < right site
    lsite, rsite = sort([site1, site2])
    @assert lsite !== rsite

    # Trace and contract sites left of left site and right of right site
    L = tr(state, 1:lsite-1)
    R = tr(state, length(state):-1:rsite+1)

    if tom
        error("Tomography twosite_reduced_density not implemented for MPO")
    else
        return L * state[lsite] * tr(state, lsite+1:rsite-1) * state[rsite] * R
    end
end

function apply_depolarizing_noise(ψ::MPO, ε::Real; _...)
    # Assume MPO site indices plev are 0 and 1
    sites = siteinds(first, ψ)
    # Multiply ε by 4/3 to agree with "make_kraus_operator"
    Ks = single_site_depolarizing_noise.(sites, ε=(4 / 3 * ε))
    return swapprime(apply.(Ks, ψ, apply_dag=true), 0 => 1, tags="Site")
end
