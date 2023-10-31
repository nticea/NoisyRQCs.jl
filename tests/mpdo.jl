
using Test
using ITensors

include("../src/MPDO.jl")
include("../src/utilities.jl")

@testset "MPDO tests" begin
    @testset "trace of pure state is 1" begin
        mps = initialize_wavefunction(L=3)
        mpdo = MPDO(mps)
        @test tr(mpdo) == 1.0
    end
end

@testset "Reduced density matrix with contraction" begin
    @testset "reduced density of product state" begin
        N = 10
        sites = siteinds("Qubit", N)
        mps = productMPS(sites, fill("0", N))
        mpdo = MPDO(mps)
        rho = twosite_reduced_density(mpdo, 3, 7)
        should_be = onehot(sites[3] => 1, sites[7] => 1, sites[3]' => 1, sites[7]' => 1)
        @test rho == should_be
    end

    @testset "reduced density of bell state" begin
        # Build bell state
        site1, site2 = siteinds("Qubit", 2)
        bell = ITensor(ComplexF64, [site1, site2])
        bell[site1=>1, site2=>1] = 1.0
        bell[site1=>2, site2=>2] = 1.0
        bell *= 1 / √2

        # Make MPS out of bell state
        U, S, V = svd(bell, site1; cutoff=0, lefttags="Link,l=1", righttags="Link,l=1")
        mps = MPS([U, S * V])
        should_be = *(density_matrix(mps)...)

        # Compute reduced density with mpdo
        rho = twosite_reduced_density(MPDO(mps), 1, 2)
        @test rho == should_be
    end

    @testset "reduced density of random state" begin
        # Build random state
        N = 5
        sites = siteinds("Qubit", N)
        psi = randomMPS(ComplexF64, sites, linkdims=2^N)
        full_density = density_matrix(psi)
        # Trace out all but 2 indices
        deltas = [δ(sites[i], sites[i]') for i in [1, 3, 5]]
        should_be = *(deltas..., full_density...)

        # Compute reduced density with mpdo
        rho = twosite_reduced_density(MPDO(psi), 2, 4)
        @test rho ≈ should_be
    end
end

@testset "Reduced density matrix with tomography" begin
    @testset "reduced density of product state" begin
        N = 10
        sites = siteinds("Qubit", N)
        mps = productMPS(sites, fill("0", N))
        mpdo = MPDO(mps)
        rho = twosite_reduced_density(mpdo, 3, 7, tom=true)
        should_be = onehot(sites[3] => 1, sites[7] => 1, sites[3]' => 1, sites[7]' => 1)
        @test rho == should_be
    end

    @testset "reduced density of bell state" begin
        # Build bell state
        site1, site2 = siteinds("Qubit", 2)
        bell = ITensor(ComplexF64, [site1, site2])
        bell[site1=>1, site2=>1] = 1.0
        bell[site1=>2, site2=>2] = 1.0
        bell *= 1 / √2

        # Make MPS out of bell state
        U, S, V = svd(bell, site1; cutoff=0, lefttags="Link,l=1", righttags="Link,l=1")
        mps = MPS([U, S * V])
        should_be = *(density_matrix(mps)...)

        # Compute reduced density with mpdo
        rho = twosite_reduced_density(MPDO(mps), 1, 2, tom=true)
        @test rho == should_be
    end

    @testset "reduced density of random state" begin
        # Build random state
        N = 5
        sites = siteinds("Qubit", N)
        psi = randomMPS(ComplexF64, sites, linkdims=2^N)
        full_density = density_matrix(psi)
        # Trace out all but 2 indices
        deltas = [δ(sites[i], sites[i]') for i in [1, 3, 5]]
        should_be = *(deltas..., full_density...)

        # Compute reduced density with mpdo
        rho = twosite_reduced_density(MPDO(psi), 2, 4, tom=true)
        @test rho ≈ should_be
    end
end;
