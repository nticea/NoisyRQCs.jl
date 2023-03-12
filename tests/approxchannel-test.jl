
using ITensors
using Test

include("../src/circuit.jl")
include("../src/utilities.jl")
include("../src/approxchannel.jl")
include("../src/kraus.jl")

@testset "approxchannel tests" begin
    @testset "array function with random mpo truncation" begin
        # 1. generate random density matrices
        nsites = 2
        bonddim = 100
        sites = siteinds("Qubit", nsites)
        psi = randomMPS(ComplexF64, sites, linkdims=bonddim)
        rho = density_matrix(psi)

        # 2. Make truncated density matrix
        nbondstrunc = 1
        centersite = nsites ÷ 2
        startsite = centersite - (nbondstrunc ÷ 2)
        siterange = startsite:(startsite+nbondstrunc)
        truncatedbonddim = 1
        trho = truncate(rho, maxdim=truncatedbonddim, site_range=siterange)

        # 3. Find approximate quantum channel
        nkraus = 6
        ρ = toarray(rho, sites, sites')
        ρ̃ = toarray(trho, sites, sites')
        Ks, optloss, initloss, iterdata, model = approxquantumchannel(ρ, ρ̃, nkraus=nkraus, silent=true)

        # Check completeness
        compl = +([Ki' * Ki for Ki in eachslice(Ks, dims=3)]...)
        @test compl ≈ I

        # Transform Kraus operator into tensor
        krausidx = Index(last(size(Ks)), KRAUS_TAG)
        K = toITensor(Ks, prime.(sites), sites, krausidx)

        # Make sure loss is the same
        approx = apply(K, rho, apply_dag=true)
        @test sum(norm.(array(*(approx...) - *(trho...))) .^ 2) ≈ optloss

        @test iscomplete(K, sites)
    end

    #TODO check approximation matches applied channel
    @testset "identity" begin end

    @testset "depolarizing" begin end

    @testset "dephasing" begin end

end;
