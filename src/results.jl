using ITensors
using Plots
using StatsBase
using CurveFit
using ITensors.HDF5

struct Results
    L::Int
    T::Int
    ε::Real
    maxdim::Int
    max_inner_dim::Union{Int,Nothing}
    state_entropy::Vector{Float64}
    operator_entanglement::Matrix{Float64}
    trace::Vector{Float64}
    logarithmic_negativity
    mutual_information
end

function entropy_dim(l::Int; d=2::Int)
    d^l
end

function plot_entropy(r::Results)
    plot_entropy(r.entropy, r.L)
end

function plot_operator_entanglement(S, L::Int; title::String="Operator Entanglement at L/2")
    mid = floor(Int, L / 2)
    if size(S)[1] != L
        S = transpose(S)
    end
    toplot = S[mid, :]
    plot(1:length(toplot), toplot, label="Operator entanglement at L/2")
    title!(title)
    xlabel!("T")
end

function saturation_value(L::Int; d::Int=2)
    # note: d is the original local site dimension 
    L_A = floor(Int, L / 2)
    L_B = L - L_A
    dimA = entropy_dim(L_A, d=d)
    dimB = entropy_dim(L_B, d=d)
    return -log.((dimA + dimB) / (dimA * dimB + 1))
end

function plot_entropy(S::Vector{Float64}, L::Int; title::String="Second Renyi Entropy at L/2", plot_ref=true)
    plot(1:length(S), S, label="Renyi entropy at L/2")
    title!(title)
    xlabel!("T")

    if plot_ref
        # plot the reference
        d = 2 # the local site dimension 
        t = collect(1:length(S))
        early_t = -log.((2 * d / (d^2 + 1)) .^ t)
        late_t = saturation_value(L, d=2)

        t_intersect = sort(findall(x -> x > late_t, early_t))[1]
        plot!(early_t[1:t_intersect], label="Early t scaling")
        hline!([late_t], label="Saturation value")
    end
end

function porter_thomas_fit(r::Results; do_fit=true)
    D = get_D(r)
    _porter_thomas_fit(r.bitdist, r.D, do_fit)
end

function porter_thomas_fit(ρ::MPO; do_fit=true, plot=true)
    # get the probability distribution over bitstrings
    bitdist = bitstring_distribution(ρ)
    D = get_D(r)
    _porter_thomas_fit(r.bitdist, D, do_fit)
end

function _porter_thomas_fit(bitdist, D::Int, do_fit)
    # histogram it 
    h = StatsBase.fit(Histogram, bitdist .* D, nbins=50)
    #h = StatsBase.normalize(h, mode=:density)
    weights, edges = h.weights, h.edges[1]
    edges = edges[1:end-1]
    weights = weights ./ sum(edges .* weights)

    # some cleanup 
    logweights = log.(weights)
    clean_idx = findall(x -> x != Inf && x != -Inf, logweights)
    logweights = logweights[clean_idx]
    edges = edges[clean_idx]

    # plot it out 
    scatter(edges, logweights)
    title!("P(Dp)")

    if do_fit
        # fit the log.(weights) to a line  
        a, b, fit_y = line_fit(edges, logweights)

        # plot it out 
        plot!(edges, fit_y, label="exponential fit, k=$b")

    end
end

function line_fit(x, y)
    a, b = linear_fit(x, y)
    fit_y = a .+ b .* x
    return a, b, fit_y
end

function square_residual(y, ỹ)
    sum((ỹ .- y) .^ 2)
end

function exponential_fit(x, y)
    a, b = exp_fit(x, y)
    fit_y = a .* exp.(b .* x)
    return a, b, fit_y
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

function bitstring_distribution(ρ::MPO)
    ψ = combine_indices(ρ)
    pdist = probability_distribution(ψ) # size N x 4 

    # From this distribution, get the distribution over bits 
    L = length(ρ)
    d = ITensors.dim(siteind(ψ, 1))
    D = d^L

    bitdist = zeros(D)
    for b in 0:(D-1)
        # convert the number to a bit representation 
        bitarr = 1 .+ reverse(digits(b, base=d, pad=L))
        probs = 1
        for l in 1:L
            probs *= pdist[l, bitarr[l]]
        end
        bitdist[b+1] = probs
    end

    return bitdist
end

function probability_distribution(ρ::MPO)
    probability_distribution(combine_indices(ρ))
end

function probability_distribution(m::MPS)
    N = length(m)
    d = ITensors.dim(siteind(m, 1))

    # if ITensors.orthocenter(m) != 1
    #   error("probability_distribution: MPS m must have orthocenter(m)==1")
    # end
    @error "There might be something funky going on here with the orthogonality centre"
    if abs(1.0 - norm(m[1])) > 1E-8
        m ./= norm(m)
        @error "probability_distribution: MPS is not normalized, norm=$(norm(m[1]))"
        #error("probability_distribution: MPS is not normalized, norm=$(norm(m[1]))")
    end

    probs = zeros(N, d)
    A = m[1]

    for j in 1:N # iterate through all the sites in the MPS 
        s = siteind(m, j) # extract each site 
        # Compute the probability of each state
        An = ITensor()
        pn = 0.0
        for n in 1:d
            projn = ITensor(s)
            projn[s=>n] = 1.0
            An = A * dag(projn)
            pn = real(scalar(dag(An) * An)) # probability of each state 
            probs[j, n] = pn
        end

        # Contract this block with the next site 
        # See "retrieving a component from an MPS/TT" from 
        # https://tensornetwork.org/mps/ 
        if j < N
            A = m[j+1] * An
            A *= (1.0 / sqrt(pn))
        end
    end
    return probs
end