import numpy as np
from ACO import population
from ACO import data_preprocessing as dpp
from ACO import ml
from ACO import t_set as ts
import pickle
import pandas as pd

# Solve the Orientiring problem with ACO methods

# load the ML model for the SVM-ACO methods
filename = 'trained_svm.sav'
model = pickle.load(open(filename, 'rb'))

# CHOOSE BETWEEN 2 ALTERNATIVES IN ORDER TO GENERATE A PROBLEM TO SOLVE
# 1) create a trial problem with chosen size n
n = 30
coord, score, T_max = dpp.generate_dataset(n)
start, end = 0, n - 1
d = pd.DataFrame({"x": coord[0, :], "y": coord[1, :], "score": score})
# generate the features
features = ts.obtain_testing_set_new(d, T_max)
# generate the p matrix
p = ml.p_matrix(np.array(features), model, n)

# 2) load a problem from the github page of the authors
# there are 100 of size 50 (n_nodes=50) and 100 of size 100 (n_nodes=100)
T_max, start, end, d = dpp.load_problem(num=3, n_nodes=50)
n = len(d)
# on the github there are also the exact solutions to 90 out of the 100 problems with n=50
# useful to compare the exact solution to the one found with ACO methods
dpp.plot_solution(num=3, start=start, end=end, n=n)
# generate the features
features = ts.obtain_testing_set(start=3, num=4, n_nodes=n)
# generate the p matrix
p = ml.p_matrix(np.array(features), model, n)

# CHOOSE HOW TO SOLVE IT

# 1) solve with AS
ant_hill = population.Population(n*100, np.array(d["score"]),
                                 np.array(d.iloc[:, :-1]), start, end, T_max, p_=p, tolerance_=20)
ant_hill.init_finalize()

# 1.1) solve with classic AS
ant_hill.ML = 0
ant_hill.LS_ = False
ant_hill.AS(10)
ant_hill.plot()  # plot the solution found

# 1.2) solve with SVM-AS
ant_hill.ML = 2
ant_hill.LS_ = False
ant_hill.AS(10)
ant_hill.plot()  # plot the solution found

# 1.3) solve with SVM-AS-LS
ant_hill.ML = 2
ant_hill.LS_ = True
ant_hill.AS(10)
ant_hill.plot()  # plot the solution found

# 2) solve with MMAS
ant_hill = population.Population(n, np.array(d["score"]),
                                 np.array(d.iloc[:, :-1]), start, end, T_max, p_=p, tolerance_=30)
ant_hill.init_finalize()

# 2.1) solve with classic MMAS
ant_hill.ML = 0
ant_hill.MMAS(100, global_=False)  # tau matrix updated by the local-best ant
ant_hill.MMAS(100, global_=True)  # tau matrix updated by the global-best ant
ant_hill.plot()  # plot the solution found

# 2.2) solve with SVM-MMAS
ant_hill.ML = 2
ant_hill.LS_ = False
ant_hill.MMAS(100, global_=False)  # tau matrix updated by the local-best ant
ant_hill.MMAS(100, global_=True)  # tau matrix updated by the global-best ant
ant_hill.plot()  # plot the solution found

# 2.3) solve with SVM-MMAS-LS
ant_hill.ML = 2
ant_hill.LS_ = True
ant_hill.MMAS(10, global_=False)  # tau matrix updated by the local-best ant
ant_hill.MMAS(10, global_=True)  # tau matrix updated by the global-best ant
ant_hill.plot()  # plot the solution found

