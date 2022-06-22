import gurobipy as gp
from gurobipy import GRB
import numpy as np
from ACO import data_preprocessing as dpp
from ACO import population
import time

# Solve the Orientiring problem with Gurobi optimization tools

# CHOOSE BETWEEN 2 ALTERNATIVES
# 1) create a trial problem with chosen size n
n = 30
coord, score, T_max = dpp.generate_dataset(n)
ant_hill_tmp = population.Population(n, score, coord.T, 0, n - 1, T_max)
ant_hill_tmp.init_finalize()
c = ant_hill_tmp.cost_matrix.copy()

# 2) load a problem from the github page of the authors
# there are 100 of size 50 (n_nodes=50) and 100 of size 100 (n_nodes=100)
T_max, start, end, d = dpp.load_problem(num=3, n_nodes=50)
n = len(d)
ant_hill_tmp = population.Population(n, np.array(d["score"]),
                                     np.array(d.iloc[:, :2]), start, end, T_max)
ant_hill_tmp.init_finalize()
score = ant_hill_tmp.gain_vector.copy()
c = ant_hill_tmp.cost_matrix.copy()

# SOLVE THE CHOSEN PROBLEM
# In the comments I will be referring to the i-th node of the variables X as "node (i+1)"
# --> example: node 1 is the 0th node of the X matrix
ORI = gp.Model()

# add the variables
X = ORI.addMVar(shape=(n, n), vtype=GRB.BINARY, name="X")
U = ORI.addMVar(shape=(n - 1,), vtype=GRB.INTEGER, name="U")

# add the constraints
ORI.addConstr(X[0, :].sum() == 1)  # node 1 is the starting node
ORI.addConstr(X[:, n - 1].sum() == 1)  # node n is the ending node
# uniqueness of trajectory
ORI.addConstrs((X[: n - 1, k].sum() == X[k, 1:].sum() for k in range(1, n - 1)))
ORI.addConstrs((X[: n - 1, k].sum() <= 1 for k in range(1, n - 1)))
# sub-tours elimination
ORI.addConstrs((U[i]-U[j]+1 <= (n - 1) * (1 - X[i + 1, j + 1]) for i in range(n - 1) for j in range(n - 1)))
ORI.addConstrs((U[i] >= 0 for i in range(n - 1)))
# max-time constraint
ORI.addConstr(gp.quicksum([c[i, j]*X[i, j] for i in range(n) for j in range(n)]) <= T_max)

# define objective function
ORI.setObjective(gp.quicksum([score[j] * X[i, j] for i in range(n - 1) for j in range(1, n)]), GRB.MAXIMIZE)

# solve
t_1 = time.time()
ORI.optimize()
t = time.time() - t_1

# PLOT THE SOLUTION FOUND
variables = ORI.getVars()
x_matrix = np.array([variables[i].X for i in range(n * n)]).reshape((n, n))
traj = dpp.from_x_to_trajectory(x_matrix)
ant_hill_tmp.trajectory_best = traj.copy()
ant_hill_tmp.y_best = ORI.getObjective().getValue()
ant_hill_tmp.plot()
