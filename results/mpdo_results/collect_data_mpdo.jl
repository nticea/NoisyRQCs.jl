## IMPORTS ##
using Pkg
Pkg.activate(joinpath(@__DIR__, "../../"))
include("../../src/MPDO.jl")
include("../../src/results.jl")

using ArgParse

ITensors.set_warn_order(50)

## PARAMETERS ##
s = ArgParseSettings()
@add_arg_table! s begin
    "L"
    arg_type = Int
    required = true
    "T"
    arg_type = Int
    required = true
    "ε"
    arg_type = Float64
    required = true
    "χ"
    arg_type = Int
    required = true
    "κ"
    arg_type = Int
    required = true
    "replica"
    arg_type = Int
    required = true
    "user"
    arg_type = String
    required = true
    "--local"
    action = :store_true
end
args = parse_args(ARGS, s)
L, T, ε, χ, κ, replica, user = args["L"], args["T"], args["ε"], args["χ"], args["κ"], args["replica"], args["user"]

stamp = "results_$(L)L_$(ε)ε_$(χ)max_outer_dim_$(κ)max_inner_dim_$(replica)replica.h5"

datapath = joinpath(@__DIR__, "data")
scratchpath = args["local"] ? joinpath(datapath, "scratch") : joinpath("/scratch/users/$(user)", "noisyRQCs")
tensorspath = joinpath(scratchpath, "folder_$(L)L_$(ε)noise_$(χ)max_outer_dim_$(κ)max_inner_dim_$(replica)replica")
mkpath(datapath)
mkpath(scratchpath)
mkpath(tensorspath)

savepath = joinpath(datapath, stamp)
benchmarkpath = joinpath(datapath, "performance_L$L.csv")
checkptpath = joinpath(scratchpath, stamp)

# Initialize the wavefunction to product state (all 0)
ψ0 = initialize_wavefunction(L=L)

# Apply the MPDO circuit
all_results = @timed apply_circuit_mpdo(ψ0, T, ε=ε, maxdim=χ,
    max_inner_dim=κ, benchmark=false, normalize_ρ=true, checkpoint_path=checkptpath, save_path=savepath, tensors_path=tensorspath)
ψ, state_entanglement, operator_entanglement, logneg, MI, trace = all_results.value

# make a results struct and save it
results = Results(0, L, T, ε, χ, κ, state_entanglement, operator_entanglement, trace, logneg, MI)
save_structs(results, savepath)

# update performance stats
df = load_performance_dataframe(benchmarkpath)
update_performance!(df, L=L, ε=ε,
    max_outer_dim=χ, max_inner_dim=κ, results=all_results)
CSV.write(benchmarkpath, df)
