# Set specific command line arguments
global ARGS = ["3", "3", "0.01", "10", "4", "2", "rdimov", "--local"]

# Include the script with modified ARGS variable
include("../results/mpdo_results/collect_data_mpdo.jl")
