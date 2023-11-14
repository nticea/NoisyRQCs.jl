

function paramstring(L::Int, T::Int, ε::Float64, χ::Int, κ::Int, type::Type; tag="")
    return "$(L)L-$(T)T-$(ε)noise-$(χ)outer-$(κ)inner-$(type)type$(isempty(tag) ? "" : "-$(tag)")"
end

function statetypestr(statedirname::String)
    pattern_type = r"([^\d-]+)type"
    match_type = match(pattern_type, statedirname)
    if match_type !== nothing
        return match_type[1]
    else
        return "MPDO"
    end
end

function build_state_params(statedirname)
    # Remove "metrics" from the end of the String
    statedirname = replace(statedirname, r"-metrics$" => "")

    # match any digits and a single decimal point
    pattern = r"(\d+(\.\d+)?)(L|T|noise|outer|inner)"
    matches = eachmatch(pattern, statedirname)
    params = Dict{String,Any}(m[3] => parse(contains(m[1], ".") ? Float64 : Int, m[1]) for m in matches)

    # New pattern for matching the type
    params["type"] = statetypestr(statedirname)

    # Include the rep
    rep = parse(Int, split(statedirname, "-")[end])
    params["rep"] = rep

    return params
end

function get_t(statefilename)
    pattern = r"state_t(\d+)"
    m = match(pattern, statefilename)
    return parse(Int, m[1])
end
