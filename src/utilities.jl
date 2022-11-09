using ITensors
using Plots 
using StatsBase 
using CurveFit



function initialize_wavefunction(;L::Int)
    @assert isodd(L) "L must be odd"
    sites = siteinds("Qubit", L)
    state_arr = ["0" for l=1:L]
    productMPS(sites,state_arr) 
end

function entanglement_entropy(ψ::MPS; b=nothing)
    if isnothing(b)
        b = ceil(Int, length(ψ)/2)
    end
    orthogonalize!(ψ, b)
    U,S,V = svd(ψ[b], (linkind(ψ, b-1), siteind(ψ,b)))
    SvN = 0.0
    for n=1:ITensors.dim(S, 1)
        p = S[n,n]^2
        SvN -= p * log(p)
    end
    return SvN 
end

function porter_thomas_fit(ρ::MPO; do_fit=true)
    # get the probability distribution over bitstrings
    bitdist = bitstring_distribution(ρ)
    
    # histogram it 
    h = StatsBase.fit(Histogram, bitdist .* length(bitdist), nbins=50)
    h = StatsBase.normalize(h, mode=:density)
    weights, edges = h.weights, h.edges[1]

    # plot it out 
    scatter(edges, log.(weights))
    title!("P(Dp)")

    # fit to exponential 
    a,b,fit_y,_ = exponential_fit(edges, weights)

    # plot it out 
    plot!(edges, fit_y, label="exponential fit, k=$b")
end

function exponential_fit(x, y)
    a,b = exp_fit(x, y)
    fit_y = a.*exp.(b.*x)
    err = square_residual(fit_y, y)
    return a,b,fit_y,err
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
    # reset_ortho_lims!(ψ) 

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

function bitstring_distribution(ρ::MPO)
    ψ = combine_indices(ρ)
    pdist = probability_distribution(ψ) # size N x 4 

    # From this distribution, get the distribution over bits 
    L = length(ρ)
    d = ITensors.dim(siteind(ψ,1))
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

    @show sum(bitdist)

    # Plot out D*bitdist 
    h = fit(Histogram, bitdist .* D)
    plot(h)

    return bitdist 
end

function probability_distribution(ρ::MPO)
    probability_distribution(combine_indices(ρ))
end

function probability_distribution(m::MPS)
    N = length(m)
    d = ITensors.dim(siteind(m,1))
  
    # if ITensors.orthocenter(m) != 1
    #   error("probability_distribution: MPS m must have orthocenter(m)==1")
    # end
    @error "There might be something funky going on here with the orthogonality centre"
    if abs(1.0 - norm(m[1])) > 1E-8
      error("probability_distribution: MPS is not normalized, norm=$(norm(m[1]))")
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
        projn[s => n] = 1.0
        An = A * dag(projn)
        pn = real(scalar(dag(An) * An)) # probability of each state 
        probs[j, n] = pn
      end

      # Contract this block with the next site 
      # See "retrieving a component from an MPS/TT" from 
      # https://tensornetwork.org/mps/ 
      if j < N
        A = m[j + 1] * An
        A *= (1.0 / sqrt(pn))
      end
    end
    return probs
  end
