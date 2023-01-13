using Convex, SCS, LinearAlgebra

n = 20
P = randn(n, n) + im * randn(n, n)
P = P * P'
Q = randn(n, n) + im * randn(n, n)
Q = Q * Q'
Z = ComplexVariable(n, n)

objectivefcn2 = 
    0.5 * real(tr(Z + Z'))


constraintfcn2 = 
    [P Z; Z' Q] ⪰ 0

problem = maximize(objectivefcn2, constraintfcn2)
solve!(problem, SCS.Optimizer; silent_solver = true)
computed_fidelity = evaluate(objective)