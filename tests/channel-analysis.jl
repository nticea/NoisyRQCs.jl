using ITensors

include("../src/channel-analysis.jl")

@testset "channel-analysis tests" begin
    # Build Kraus operator for testing
    nsites = 1
    nkraus = (2^nsites)^2
    bonddim = 8
    sites = siteinds("S=1/2", nsites)
    krausidx = Index(nkraus, "Kraus")
    K = randomITensor(ComplexF64, sites, prime(sites), krausidx)

    Cs, basis, labels = paulidecomp(K, sites)

    # Check that we can sreconstruct the Kraus tensor
    reconstruction = sum(Cs .* basis)
    @test reconstruction ≈ K

    norms = frobneiusnorm(K, krausidx)
end;
s
