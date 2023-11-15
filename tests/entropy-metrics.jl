using Test
using ITensors

include("../src/paulis.jl")
include("../src/mpdo.jl")
include("../src/entropy-metrics.jl")

function belldensity()
    bell = ITensor(ComplexF64, [site1, site2])
    bell[site1=>1, site2=>1] = 1.0
    bell[site1=>2, site2=>2] = 1.0
    bell *= 1 / √2
    rho = bell * bell'
    return rho, (site1, site2)
end

@testset "Logarithmic negativity" begin
    @testset "Logarithmic negativity of bell state is 1" begin
        b, sites = belldensity()
        @test logarithmic_negativity(b, [sites[1]]) ≈ 1.0
        @test logarithmic_negativity(b, [sites[2]]) ≈ 1.0
    end

    @testset "Logarithmic negativity of mixed state is 0" begin
        site1, site2 = siteinds("Qubit", 2)
        p1 = buildpaulibasis(site1)
        p2 = buildpaulibasis(site2)
        mixed = 1 / 4 * p1[1] * p2[2]
        @test logarithmic_negativity(mixed, [site1]) ≈ 0.0
    end

    @testset "Logarithmic negativity of product state is 0" begin
        site1, site2 = siteinds("Qubit", 2)
        prodmps = productMPS([site1, site2], ["0", "1"])
        rho = prod(outer(prodmps, prodmps'))
        @test logarithmic_negativity(rho, [site1]) ≈ 0.0
    end

    @testset "Logarithmic negativity of Werner state is 0" begin
        s1, s2, s3 = siteinds("Qubit", 3)
        w = ITensor(ComplexF64, [s1, s2, s3])
        w[s1=>2, s2=>1, s3=>1] = 1.0
        w[s1=>1, s2=>2, s3=>1] = 1.0
        w[s1=>1, s2=>1, s3=>2] = 1.0
        w *= 1 / √3
        rho = w * w'
        pt = rho * δ(s3, s3')
        @test logarithmic_negativity(pt, [s1]) ≈ log2((2 + √5) / 3)
    end
end
