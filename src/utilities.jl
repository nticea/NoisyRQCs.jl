using ITensors
using Plots 
using StatsBase 
using CurveFit
using ITensors: linkinds
import Base.isapprox


struct Results
    L::Int
    T::Int 
    ρ::Union{MPO,Int64}
    bitdist::Vector{Float64}
    state_entropy::Vector{Float64}
    operator_entanglement::Matrix{Float64}
    trace::Vector{Float64}
end

function initialize_wavefunction(;L::Int)
    @assert isodd(L) "L must be odd"
    sites = siteinds("Qubit", L)
    state_arr = ["0" for l=1:L]
    productMPS(sites,state_arr) 
end

function entropy_dim(l::Int; d=2::Int)
    d^l
end

function plot_entropy(r::Results)
    plot_entropy(r.entropy, r.L)
end

function saturation_value(L::Int; d::Int=2)
    # note: d is the original local site dimension 
    L_A = floor(Int, L/2)
    L_B = L - L_A 
    dimA = entropy_dim(L_A,d=d)
    dimB = entropy_dim(L_B,d=d)
    return -log.((dimA + dimB)/(dimA * dimB + 1))
end

function plot_entropy(S::Vector{Float64}, L::Int)
    plot(1:length(S), S, label="Renyi entropy")
    title!("Second Renyi Entropy at L/2")
    xlabel!("T")

    # plot the reference
    d = 2 # the local site dimension 
    t = collect(1:length(S))
    early_t = -log.((2*d/(d^2+1)) .^ t)
    late_t = saturation_value(L, d=2)

    t_intersect = sort(findall(x -> x>late_t, early_t))[1]
    plot!(early_t[1:t_intersect], label="Early t scaling")
    hline!([late_t], label="Saturation value")
end

function entanglement_entropy(ψ::MPS; b=nothing)
    if isnothing(b)
        b = floor(Int, length(ψ)/2)
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

function second_Renyi_entropy(ρ)
    return -log(tr(apply(ρ, ρ)))
end

findnearest(A,x) = argmin(abs.(A .- x))

"""
physical_indices(ψ::MPS, idxlist::Vector{Int}, tag::String)
    Given an INTEGER list of desired indices, 
    return a list of the corresponding PHYSICAL Index (struct) of the MPS  
"""
function physical_indices(ψ::Union{MPS,MPO}, sitelist::Vector{Int}; tag::String="Site")
    [getfirst(x -> hastags(x, tag), inds(ψ[s])) for s in sitelist]
end

function physical_indices(ψ::Union{MPS,MPO}; tag::String="Site")
    [getfirst(x -> hastags(x, tag), inds(ψs)) for ψs in ψ]
end

function physical_indices(ψ::ITensor; tag::String="Site")
    getfirst(x -> hastags(x, tag), inds(ψ))
end

# here we are KEEPING the indices in indslist 
function reduced_density_matrix(ρ::MPO, indslist::Vector{Int})
    # the indices to keep must be a continuous chunk
    sort!(indslist)
    @assert indslist == collect(indslist[begin]:indslist[end])
    L = length(ρ)

    # first, trace out the indices on the LHS  
    linds = collect(1:indslist[begin]-1)
    if length(linds) > 0
        ρ = partial_trace(ρ, linds, "left")
    end
    
    # now trace out the indices on the RHS 
    rinds = collect(indslist[end]+1:L) .- length(linds)
    if length(rinds) > 0
        ρ = partial_trace(ρ, rinds, "right")
    end

    return ρ
end

# here we are TRACING OUT the indices in indslist 
function partial_trace(ρ::MPO, indslist::Vector{Int}, side::String)
    ρ = copy(ρ)
    s = physical_indices(ρ) # these are the physical sites 

    if side=="left"
        border_idx = indslist[end]+1
    elseif side=="right"
        border_idx = indslist[begin]-1
    else
        @error side*" is not recognized"
    end

    orthogonalize!(ρ, border_idx)

    # trace out the indices in indslist
    for i in indslist
        ρ[i] = ρ[i]*delta(s[i],prime(s[i]))
    end

    # contract the indices in indslist
    L = ITensor(1.0)
    for i in indslist
        L *= ρ[i]
    end

    # mutliply this into the remaining ρ
    ρ[border_idx] *= L 

    to_keep = setdiff(collect(1:length(ρ)), indslist)
    ρ_new = MPO(ρ[to_keep])

    return ρ_new 
end

function get_D(ρ::MPO)
    L = length(ρ)
    d = ITensors.dim(siteind(ρ,1))
    D = d^L
    return D 
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
    clean_idx = findall(x -> x!= Inf && x != -Inf, logweights)
    logweights = logweights[clean_idx]
    edges = edges[clean_idx]

    # plot it out 
    scatter(edges, logweights)
    title!("P(Dp)")

    if do_fit
        # fit the log.(weights) to a line  
        a,b,fit_y = line_fit(edges, logweights)

        # plot it out 
        plot!(edges, fit_y, label="exponential fit, k=$b")

    end
end

function line_fit(x, y)
    a,b = linear_fit(x, y)
    fit_y = a .+ b .* x
    return a,b,fit_y
end

function square_residual(y, ỹ)
    sum((ỹ .- y).^2)
end

function exponential_fit(x, y)
    a,b = exp_fit(x, y)
    fit_y = a.*exp.(b.*x)
    return a,b,fit_y
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

    return bitdist 
end

function probability_distribution(ρ::MPO)
    probability_distribution(combine_indices(ρ))
end

"""
physical_indices(ψ::MPS, idxlist::Vector{Int}, tag::String)
    Given an INTEGER list of desired indices, 
    return a list of the corresponding PHYSICAL Index (struct) of the MPS  
"""
function physical_indices(ψ::Union{MPS,MPO}, sitelist::Vector{Int}; tag::String="Site")
    [getfirst(x -> hastags(x, tag), inds(ψ[s])) for s in sitelist]
end

function physical_indices(ψ::Union{MPS,MPO}; tag::String="Site")
    [getfirst(x -> hastags(x, tag), inds(ψs)) for ψs in ψ]
end

function get_plev_inds(T::ITensor, lev::Int)
    inds(T)[findall(x -> plev(x)==lev, inds(T))]
end

function primed_inds(T::ITensor)
    return get_plev_inds(T, 1)
end

function unprimed_inds(T::ITensor)
    return get_plev_inds(T, 0)
end

function taginds(T::ITensor, tag::String)
    inds(T)[findall(x -> hastags(x, tag), inds(T))]
end

function tag_and_plev(T::ITensor; tag::String, lev::Int)
    tinds = taginds(T, tag)
    tinds[findall(x -> plev(x)==lev, tinds)]
end

function linkindT(T::ITensor)
    taginds(T, "Link")
end

function siteindT(T::ITensor)
    taginds(T, "Site")
end

function isapprox(T1::ITensor, T2::ITensor; atol=1e-6)
    # permute indices to match up 
    if inds(T1) != inds(T2)
        T1 = permute(T1, inds(T2))
    end

    # extract elements and compare
    T1_arr = [array(T1)...]
    T2_arr = [array(T2)...]

    for (t1,t2) in zip(T1_arr,T2_arr)
        @assert isapprox.(t1, t2, atol=atol)
    end
    return true 
end

function probability_distribution(m::MPS)
    N = length(m)
    d = ITensors.dim(siteind(m,1))
  
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

  using ITensors.HDF5

function save_structs(struc, path::String)
    function Name(arg)
        string(arg)
    end
    fnames = fieldnames(typeof(struc))
    for fn in fnames 
        n = Name(fn)
        d = getfield(struc, fn)

        # If the file already exists, then we either append to it or overwrite 
        if isfile(path)
            h5open(path, "r+") do file
                if haskey(file, n) #if key already exists, we want to rewrite 
                    delete_object(file, n)
                    write(file, n, d)
                else
                    write(file, n, d) 
                end
            end
        else # If the file does not exist, create it 
            h5open(path, "w") do file
                write(file, n, d) 
            end
        end
    end
end

function load_results(loadpath::String; load_state=false)
    f = h5open(loadpath,"r")
    if load_state
        ρ = read(f, "ρ", MPO)
    else
        ρ = 0 
    end
    d = read(f)
    return Results(d["L"], d["T"], ρ, d["bitdist"], d["state_entropy"], 
            d["operator_entanglement"], d["trace"])
end