@kwdef struct SbatchParams
    jobname::String
    partition::String = "owners,simes"
    time::String = "3:00:00"
    nodes::Int = 1
    ntasks::Int = 1
    cpuspertask::Int = 8
    memG::Int = 256
    user::String
end

function submitjob(scriptpath, args, params::SbatchParams)
    slurmdir = joinpath(@__DIR__, "..", "slurm")
    outdir = joinpath(slurmdir, "out")
    sbatchdir = joinpath(slurmdir, "sbatch")
    mkpath(outdir)
    mkpath(sbatchdir)

    sbatchfilepath = joinpath(sbatchdir, "$(params.jobname).sbatch")
    sbatchstr = build_sbatch(scriptpath, outdir, args, params)
    open(sbatchfilepath, "w") do io
        write(io, sbatchstr)
    end
    run(`sbatch $sbatchfilepath`)
end

function build_sbatch(scriptpath::String, outdir::String, args::Array{String}, params::SbatchParams)
    return """#!/bin/bash
    #SBATCH --job-name=$(params.jobname)
    #SBATCH --partition=$(params.partition)
    #SBATCH --time=$(params.time)
    #SBATCH --nodes=$(params.nodes)
    #SBATCH --ntasks=$(params.ntasks)
    #SBATCH --cpus-per-task=$(params.cpuspertask)
    #SBATCH --mem=$(params.memG)G
    #SBATCH --mail-type=BEGIN,FAIL,END
    #SBATCH --mail-user=$(params.user)@stanford.edu
    #SBATCH --output=$(joinpath(outdir, "$(params.jobname)_output.txt"))
    #SBATCH --error=$(joinpath(outdir, "$(params.jobname)_error.txt"))
    #SBATCH --open-mode=append

    # load Julia module
    ml julia

    # run the script
    julia $scriptpath $(join(args, " "))"""
end
