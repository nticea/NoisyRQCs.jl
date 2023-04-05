## IMPORTS ##
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
include("../src/circuit.jl")
include("../src/results.jl")
using Random

ITensors.set_warn_order(50)
Random.seed!(123456)

## PARAMETERS ## 
L = 9
T = 4
maxdims = nothing
random_type = "Haar"

# Initialize the wavefunction to product state (all 0)
ψ0 = initialize_wavefunction(L=L)
sites = siteinds(ψ0)

# Make the density matrix 
global ρ = density_matrix(copy(ψ0))

function my_svd(A::ITensor, Linds...)

    # Keyword argument deprecations
    #if haskey(kwargs, :utags) || haskey(kwargs, :vtags)
    #  @warn "Keyword arguments `utags` and `vtags` are deprecated in favor of `leftags` and `righttags`."
    #end

    Lis = commoninds(A, Linds)
    Ris = uniqueinds(A, Lis)

    CL = combiner(Lis...)
    CR = combiner(Ris...)

    AC = A * CR * CL

    cL = combinedind(CL)
    cR = combinedind(CR)

    @show inds(AC)
    @show cL
    @show cR

    if inds(AC) != (cL, cR)
        AC = permute(AC, cL, cR)
    end

    F = LinearAlgebra.svd(array(AC))
    return F.S
end

function my_svn(T::ITensor)
    sites = tag_and_plev(T; tag="Site", lev=0)
    S = my_svd(T, sites...)
    SvN = 0.0
    for p in S
        if p != 0
            SvN -= p * log(p)
        end
    end
    return SvN
end

for t in 1:2:T
    A = 1
    for B in collect(2:L)
        @show A, B

        ρA, ρB, ρAB = twosite_reduced_density_matrix(ρ, A, B)

        SA = my_svn(ρA)
        SB = my_svn(ρB)
        SAB = my_svn(ρAB)
        MI = SA + SB - SAB

        @show SA, SB, SAB
        @show MI

        # ρA = permute_to_svd_form(ρA)
        # ρB = permute_to_svd_form(ρB)
        # ρAB = permute_to_svd_form(ρAB)

        # @show array(ρA)
        # @show array(ρB)
        # @show array(ρAB)
    end

    # At each time point, make a layer of random unitary gates 
    unitary_gates = unitary_layer(sites, t, random_type)

    # Now apply the gates to the wavefunction (alternate odd and even) 
    # for u in unitary_gates
    #     global ρ = apply_twosite_gate(ρ, u, maxdim=maxdims)
    # end
    global ρ = apply_twosite_gate(ρ, unitary_gates[1], maxdim=maxdims)
    global ρ = apply_twosite_gate(ρ, unitary_gates[2], maxdim=maxdims)
end