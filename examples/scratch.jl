using Convex, SCS, LinearAlgebra

n = 20
P = randn(n, n) + im * randn(n, n)
P = P * P'
Q = randn(n, n) + im * randn(n, n)
Q = Q * Q'
Z = ComplexVariable(n, n)
objective = 0.5 * real(tr(Z + Z'))
constraint = [P Z; Z' Q] ⪰ 0
problem = maximize(objective, constraint)
solve!(problem, SCS.Optimizer; silent_solver = true)
computed_fidelity = evaluate(objective)

@show typeof(objective)
@show typeof(constraint)