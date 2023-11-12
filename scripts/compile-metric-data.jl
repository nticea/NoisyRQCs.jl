using Pkg
Pkg.activate(joinpath(@__DIR__, "../"))

using CSV
using DataFrames
using Glob

include("../src/file-parsing.jl")

# Define the path to the data directory
datadir = joinpath(@__DIR__, "..", "data")

# Get a list of all state directories inside the data directory
statefiles = glob(joinpath("states*", "state_t*.csv"), datadir)
println("Compiling $(length(statefiles)) data files...")

# Initialize an empty DataFrame to store the combined data
combined_filename = "combined.csv"
combined_path = joinpath(datadir, combined_filename)
alldfs = DataFrame[]

# Loop through each state directory
for statefile in statefiles
    # Get the state parameters from the directory name
    statedirname = splitpath(dirname(statefile))[end]
    stateparams = build_state_params(statedirname)

    # Get the t (timestep) from the state filename
    statefilename = basename(statefile)
    t = get_t(statefilename)

    statedata = CSV.File(statefile) |> DataFrame
    # Add parameters and t to record
    insertcols!(statedata, "t" => t, stateparams...)

    push!(alldfs, statedata)
end

combined = vcat(alldfs...)

# Write the combined data to a master CSV file
CSV.write(combined_path, combined)

println("Done!\n")
