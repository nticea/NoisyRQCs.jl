using ITensors
using ITensors.HDF5
using LinearAlgebra

include("circuit_elements.jl")
include("utilities.jl")
include("MPDO.jl")
include("results.jl")

function evolve_state(L::Int, T::Int, ε::Float64, χ::Int, κ::Int, savedir::String; tag, save_increment::Int=1)
    # Build path to directory with states
    statedirname = build_state_dirname(L, T, ε, χ, κ, tag=tag)
    statedir = joinpath(savedir, statedirname)

    # Check if any states are saved in savedir, which may not exist
    is_saved = isdir(statedir) && !isempty(readdir(statedir))
    if is_saved
        # Load saved state and continue evolution
        println("Loading saved state...")
        statefilename, t = get_latest_state_filename(statedir)
        statefile = joinpath(statedir, statefilename)
        state = load_state(statefile)
    else
        # Make a new state directory
        mkpath(statedir)

        # Build new state and start evolution
        println("Building initial state...")
        state_mps = initialize_wavefunction(L=L)
        state = build_mpdo(state_mps)
        t = 0
    end
    evolve_state(state, t, T, ε, χ, κ, statedir, save_increment=save_increment)
end

function evolve_state(state, t::Int, T::Int, ε::Float64, χ::Int, κ::Int, savedir::String; save_increment::Int=1)
    while t < T
        t += 1
        println("Evolving state to t=$(t)...")
        state = apply_timestep(state, t, ε, χ, κ)

        # Save state every save_increment time steps
        if (mod1(t, save_increment) == 1) || (t == T) # save last state as well
            filename = "state_t$(t)"
            save_state(savedir, filename, state)
        end
    end
    println("Finished time evolution!")
end

function apply_timestep(state, t::Int, ε::Float64, χ::Int, κ::Int; random_type::String="Haar")
    # prepare the noise gates
    sites = siteinds(state)

    # Apply a layer of unitary gates
    unitary_gates = unitary_layer(sites, t, random_type)
    for u in unitary_gates
        state = apply_twosite_gate(state, u, maxdim=χ)
    end

    # Apply the noise layers
    Ks = make_kraus_operators(sites, ε)
    return apply_noise_mpdo(state, Ks, inner_dim=κ)
end

## Saving and loading states

STATE_KEY = "state"

function build_state_dirname(L::Int, T::Int, ε::Float64, χ::Int, κ::Int; tag="")
    return "states-$(L)L-$(T)T-$(ε)noise-$(χ)outer-$(κ)inner$(isempty(tag) ? "" : "-$(tag)")"
end

function save_state(dir::String, filename::String, state)
    filename_with_ext = filename * ".h5"
    path = joinpath(dir, filename_with_ext)
    println("Saving state at $(path)...")
    h5open(path, "w") do file
        file[STATE_KEY] = state
    end
end

function load_state(path::String)
    file = h5open(path, "r")
    state = read(file, STATE_KEY, MPS)
    close(file)
    return state
end

function get_state_file_t(filepath::String)
    filename_without_ext, _ = splitext(basename(filepath))
    return parse(Int, filename_without_ext[end])
end

function get_latest_state_filename(dir::String)::Tuple{String,Int}
    files = readdir(dir)
    ts = get_state_file_t.(files)
    latest_idx = argmax(ts)
    return files[latest_idx], ts[latest_idx]
end
