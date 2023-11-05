using BenchmarkTools
using Plots

include("../src/MPDO.jl")

function benchmark_twosite_reduced_density(L, linkdims_range, tom)
    runtimes = []
    memory_footprints = []

    for linkdims in linkdims_range
        # Generate MPDO state with given linkdims
        sites = siteinds("Qubit", L)
        mps = randomMPS(ComplexF64, sites, linkdims=linkdims)
        mpdo = MPDO(mps)

        # Benchmark twosite_reduced_density function
        result = @benchmark twosite_reduced_density($mpdo, 1, 2, tom=$tom)

        push!(runtimes, mean(result.times) / 1e9)  # Convert to seconds
        push!(memory_footprints, mean(result.memory) / 1024)  # Convert to kilobytes
    end

    return runtimes, memory_footprints
end

L = 7
linkdims_range = [8, 12, 16]

# Benchmark with tom=true
runtimes_tom, memory_footprints_tom = benchmark_twosite_reduced_density(L, linkdims_range, true)

print(runtimes_tom, memory_footprints_tom)

# Benchmark with tom=false
runtimes_no_tom, memory_footprints_no_tom = benchmark_twosite_reduced_density(L, linkdims_range, false)

print(runtimes_no_tom, memory_footprints_no_tom)

# Plotting
plot(linkdims_range, runtimes_tom, label="tom=true", marker=:circle, xlabel="Linkdims", ylabel="Runtime (s)", title="Benchmark Results")
plot!(linkdims_range, runtimes_no_tom, label="tom=false", marker=:circle)
plot!(linkdims_range, memory_footprints_tom, label="Memory (tom=true)", marker=:circle, secondary=true)
plot!(linkdims_range, memory_footprints_no_tom, label="Memory (tom=false)", marker=:circle, secondary=true, xlabel="Linkdims", ylabel="Memory (KB)")
