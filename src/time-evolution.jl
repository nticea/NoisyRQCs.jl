using ITensors
using ITensors.HDF5
using LinearAlgebra

include("circuit_elements.jl")
include("utilities.jl")
include("mpdo.jl")
include("mpo.jl")
include("results.jl")
include("file-parsing.jl")
include("kraus.jl")

function typefromstr(str)::Type
    if str == "MPS"
        return MPS
    elseif str == "MPO"
        return MPO
    elseif str == "MPDO"
        return MPDO
    else
        error("Invalid state type string: $str")
    end
end

function initstate(L)::MPS
    @assert isodd(L) "L must be odd"
    sites = siteinds("Qubit", L)
    states = fill("0", L)
    productMPS(sites, states)
end

initstate(L, ::Type{MPDO})::MPDO = MPDO(initstate(L))
initstate(L, ::Type{MPS})::MPS = initstate(L)
initstate(L, ::Type{MPO})::MPO = density(initstate(L))

function evolve_state(L::Int, T::Int, ε::Float64, χ::Int, κ::Int, savedir::String; tag, save_increment::Int=1, type::Type)
    # Build path to directory with states
    statedirname = build_state_dirname(L, T, ε, χ, κ, type, tag=tag)
    statedir = joinpath(savedir, statedirname)

    # Check if any states are saved in savedir, which may not exist
    is_saved = isdir(statedir) && !isempty(readdir(statedir))
    if is_saved
        # Load saved state and continue evolution
        println("Loading saved state...")
        statefilename, t = get_latest_state_filename(statedir)
        statefile = joinpath(statedir, statefilename)
        state = load_state(statefile, type)
    else
        # Make a new state directory
        mkpath(statedir)

        # Build new state and start evolution
        println("Building initial state...")
        state = initstate(L, type)
        t = 0
    end
    evolve_state(state, t, T, ε, χ, κ, statedir, save_increment=save_increment)
end

STATE_FILE_PREFIX = "state_t"

function evolve_state(state, t::Int, T::Int, ε::Float64, χ::Int, κ::Int, savedir::String; save_increment::Int=1)
    while t < T
        t += 1
        println("Evolving state to t=$(t)...")
        state = apply_timestep(state, t, ε, χ, κ)

        # Save state every save_increment time steps
        if (mod1(t, save_increment) == 1) || (t == T) # save last state as well
            filename = "$(STATE_FILE_PREFIX)$(t)"
            save_state(savedir, filename, state)
        end
    end
    println("Finished time evolution!")
end

function apply_timestep(state, t::Int, ε::Float64, χ::Int, κ::Int; random_type::String="Haar")
    # Apply a layer of unitary gates
    sites = siteinds(first, state)
    unitary_gates = unitary_layer(sites, t, random_type)
    for u in unitary_gates
        state = apply_twosite_gate(state, u, maxdim=χ)
    end

    # Apply the noise layers
    if ε > 0
        state = apply_depolarizing_noise(state, ε, inner_dim=κ)
    end

    return state
end

## Saving and loading states

STATE_KEY = "state"

function build_state_dirname(L::Int, T::Int, ε::Float64, χ::Int, κ::Int, type::Type; tag="")
    return "states-$(paramstring(L, T, ε, χ, κ, type; tag))"
end

function save_state(dir::String, filename::String, state)
    filename_with_ext = filename * ".h5"
    path = joinpath(dir, filename_with_ext)
    println("Saving state at $(path)...")
    h5open(path, "w") do file
        write(file, STATE_KEY, state)
    end
end

function load_state(path::String, type::Type)
    file = h5open(path, "r")
    state = read(file, STATE_KEY, type)
    close(file)
    return state
end

function get_state_file_t(filepath::String)
    filename_without_ext, _ = splitext(basename(filepath))
    t_str = split(filename_without_ext, STATE_FILE_PREFIX)[end]
    return parse(Int, t_str)
end

function get_latest_state_filename(dir::String)::Tuple{String,Int}
    files = readdir(dir)
    ts = get_state_file_t.(files)
    latest_idx = argmax(ts)
    return files[latest_idx], ts[latest_idx]
end
