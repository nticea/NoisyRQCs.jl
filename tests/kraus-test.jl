
using ITensors

include("../src/circuit.jl")
include("../src/utilities.jl")
include("../src/approxchannel.jl")
include("../src/kraus.jl")

@testset "kraus tests" begin
    @testset "kraus validity" begin
        #TODO test isvalidkraus() with identity and incomplete Kraus
    end

    @testset "generated channels" begin
        # Test generated channels
        sites = siteinds("Qubit", 3)

        K = depolarizing_noise(sites, 0.5)
        @test isvalidkraus(K, sites)

        K = dephasing_noise(sites, 0.5)
        @test isvalidkraus(K, sites)
    end
end;
