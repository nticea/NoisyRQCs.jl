
function build_state_params(statedirname)
    # match any digits and a single decimal point
    pattern = r"(\d+(\.\d+)?)(L|T|noise|outer|inner)"
    matches = eachmatch(pattern, statedirname)
    params = Dict(m[3] => parse(contains(m[1], ".") ? Float64 : Int, m[1]) for m in matches)

    # Include the rep
    rep = parse(Int, split(statedirname, "-")[end-1])
    params["rep"] = rep

    return params
end

function get_t(statefilename)
    pattern = r"state_t(\d+)"
    m = match(pattern, statefilename)
    return parse(Int, m[1])
end
