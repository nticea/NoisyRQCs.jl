struct RunParams
    L::Int
    T::Int
    ε::Real
    χ::Int
    κ::Int
    replica::Int
end

function submit_job(params::RunParams, filepath, dirpath, job_prefix; nodes=1, ntasks=1, cpus_per_task=8, mem=256, partition="owners")
    outpath = joinpath(dirpath, "out")
    slurmpath = joinpath(dirpath, "slurmfiles")
    mkpath(outpath)
    mkpath(slurmpath)

    name = "$(params.L)L_$(params.ε)noise_$(params.χ)outerdim_$(params.κ)innerdim_$(params.replica)rep"

    filestr = """#!/bin/bash
    #SBATCH --job-name=$(job_prefix*"_"*name)
    #SBATCH --partition=$partition
    #SBATCH --time=48:00:00
    #SBATCH --nodes=$nodes
    #SBATCH --ntasks=$ntasks
    #SBATCH --cpus-per-task=$cpus_per_task
    #SBATCH --mem=$(mem)G
    #SBATCH --mail-type=BEGIN,FAIL,END
    #SBATCH --mail-user=nticea@stanford.edu
    #SBATCH --output=$outpath/$(job_prefix*"_"*name)_output.txt
    #SBATCH --error=$outpath/$(job_prefix*"_"*name)_error.txt
    #SBATCH --open-mode=append

    # load Julia module
    ml julia

    # NO MULTITHREADING!

    # run the script
    julia $filepath $(params.L) $(params.T) $(params.ε) $(params.χ) $(params.κ) $(params.replica)"""

    open("$slurmpath/$(name).slurm", "w") do io
        write(io, filestr)
    end
    run(`sbatch $(slurmpath)/$(name).slurm`)
end