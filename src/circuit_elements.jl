using ITensors
using Distributions
using LinearAlgebra
using StatsBase 

include("utilities.jl")

"""
This function generates a Haar random unitary of dimension DxD
see: https://colab.research.google.com/drive/1JRvzfG2SzNel4u80D2rnzO8xG5kDQ9aY?authuser=1#scrollTo=yyFzzZdE7s1u
"""
function gen_Haar(N)
    x = (rand(N,N) + rand(N,N)*im) / sqrt(2)
    f = qr(x)
    diagR = sign.(real(diag(f.R)))
    diagR[diagR.==0] .= 1
    diagRm = diagm(diagR)
    u = f.Q * diagRm
    return u
end

"""
This function makes a gate that is an ITensor object acting on the indices ind1 and ind2 
"""
function make_unitary_gate(ind1, ind2, random_type::String)
    D = ITensors.dim(ind1)
    @assert D==ITensors.dim(ind2)

    if random_type=="Haar"
        randU_elems = gen_Haar(D*D)
        U = ITensor(randU_elems, ind1, ind2, prime(ind1, 2), prime(ind2, 2))
        Udag = prime(dag(U))

        return U * Udag      
    else
        @assert false "Only Haar distribution implemented thus far"
    end
end

function make_kraus_gate(s, ε::Real)
    D = ITensors.dim(s)

    # Make the kraus operators
    Id = sqrt(1-ε) * Matrix(I, D, D)
    σx = sqrt(ε/3) * [0.0 1.0 
          1.0 0.0] 
    σy = sqrt(ε/3) * [0.0 -1.0im 
          -1.0im 0.0]
    σz = sqrt(ε/3) * [1.0 0.0 
          0.0 -1.0]

    # Stack them together 
    K_elems = cat(Id, σx, σy, σz, dims=3)

    # Turn this into an ITensor with the appropriate indices 
    sum_idx = Index(4, tags="sum")
    K = ITensor(K_elems, s, prime(s, 2), sum_idx, tags="Kraus")
    Kdag = prime(dag(K))

    return K * Kdag * delta(sum_idx, prime(sum_idx))
end

"""
Apply a quantum channel ("gate") to a density matrix 
"""
function apply_twosite_gate(ρ::MPO, G::ITensor; maxdim=nothing)
    ρ̃ = copy(ρ)

    # Extract the common indices where we will be applying the channel 
    c = findall(x -> hascommoninds(G, ρ[x]), collect(1:length(ρ)))
    @assert length(c) == 2
    c1, c2 = c

    # Orthogonalize the MPS around this site 
    orthogonalize!(ρ,c1)

    # Apply the gate 
    wf = (ρ̃[c1] * ρ̃[c2]) * G

    # Lower the prime level by 1 to get back to what we originally had 
    wf = replaceprime(wf, 3 => 1)
    wf = replaceprime(wf, 2 => 0)

    # SVD the resulting tensor 
    inds3 = uniqueinds(ρ̃[c1], ρ̃[c2])
    # If maxdim is nothing, then implement no truncation cutoff
    if isnothing(maxdim)
        U,S,V = ITensors.svd(wf,inds3,cutoff=0)
    else
        U,S,V = ITensors.svd(wf,inds3,maxdim=maxdim)
    end

    # Update the original MPO 
    ρ̃[c1] = U
    ρ̃[c2] = S*V

    return ρ̃
end

function apply_onesite_gate(ρ::MPO, G::ITensor)
    ρ̃ = copy(ρ)

    # extract the common indices where we will be applying the channel 
    c = findall(x -> hascommoninds(G, ρ[x]), collect(1:length(ρ)))
    @assert length(c) == 1
    c = c[1]

    # Orthogonalize around this site 
    orthogonalize!(ρ,c)

    # Apply the gate 
    wf = ρ̃[c] * G

    # Lower the prime level by 1 to get back to what we originally had 
    wf = replaceprime(wf, 3 => 1)
    wf = replaceprime(wf, 2 => 0)

    # Update the MPO 
    ρ̃[c] = wf

    return ρ̃
end
