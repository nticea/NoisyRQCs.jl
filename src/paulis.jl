using ITensors

const Id = [1.0 0
    0.0 1.0]
const σx = [0.0 1.0
    1.0 0.0]
const σy = [0.0 -1.0im
    1.0im 0.0]
const σz = [1.0 0.0
    0.0 -1.0]
const paulis = [Id, σx, σy, σz]

buildpaulibasis(site) = [ITensor(pauli, site, site') for pauli in paulis]

function paulibasislabels(n::Int)
    sitelabels = [
        [l for l in ["I", "x", "y", "z"]]
        for _ in 1:n
    ]
    return [*(ops...) for ops in Iterators.product(sitelabels...)]
end
