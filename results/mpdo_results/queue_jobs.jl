## IMPORTS ##
using Pkg
Pkg.activate(joinpath(@__DIR__, "../../"))

nreps = 5 # number of replicas 
L = 15
T = 20
ε = 1e-4
χs = [30, 50, 80]
κs = [1, 2, 4]

for r in 1:nreps
    for χ in 1:χs
        for κ in 1:κs
            # make the model parameters

            # pass them into the job dispatch function 
        end
    end
end