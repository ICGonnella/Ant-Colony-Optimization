import numpy as np
import pandas as pd
from ACO import population
import os
from ACO import data_preprocessing as dpp


def obtain_training_set(start=0, num=100, n_nodes=50):
    training_dataset = pd.DataFrame()
    for i in range(start, num):
        if os.path.exists( f"./MLACO/Datasets/random{n_nodes}/0{i}.opt.tour") or os.path.exists(
                f"./MLACO/Datasets/random{n_nodes}/00{i}.opt.tour"):
            T_max, start, end, d = dpp.load_problem(i)
            n = len(d)
            ant_hill = population.Population(n * 100, np.array(d["score"]),
                                             np.array(d.iloc[:, :-1]), start, end,
                                             T_max, 1, 100, 0.05, 0.8, tolerance_=15)
            ant_hill.init_finalize()
            pop = ant_hill.features()
            pop.index = [i for i in range(len(pop))]
            training = pd.concat([pop,
                                  pd.DataFrame({"label": [
                                      i for i in dpp.labels(dpp.load_solution(i, start,
                                                                              end, n),
                                                            n).reshape(-1)]})], axis=1)
            training_dataset = pd.concat([training_dataset, training], axis=0)
    return training_dataset


def obtain_testing_set(start=0, num=100, n_nodes=50):
    testing_dataset = pd.DataFrame()
    for i in range(start, num):
        if os.path.exists( f"./MLACO/Datasets/random{n_nodes}/0{i}.op") or os.path.exists(
                f"./MLACO/Datasets/random{n_nodes}/00{i}.op"):
            T_max, start, end, d = dpp.load_problem(i, n_nodes)
            n = len(d)
            ant_hill = population.Population(n * 100, np.array(d["score"]),
                                             np.array(d.iloc[:, :-1]), start, end,
                                             T_max, 1, 100, 0.05, 0.8, tolerance_=15)
            ant_hill.init_finalize()
            pop = ant_hill.features()
            pop.index = [i for i in range(len(pop))]
            testing_dataset = pd.concat([testing_dataset, pop], axis=0)
    return testing_dataset


def obtain_testing_set_new(d, T_max):
    testing_dataset = pd.DataFrame()
    n = len(d)
    start, end = 0, n - 1
    ant_hill = population.Population(n * 100, np.array(d["score"]),
                                        np.array(d.iloc[:, :-1]), start, end,
                                        T_max, 1, 100, 0.05, 0.8, tolerance_=15)
    ant_hill.init_finalize()
    pop = ant_hill.features()
    pop.index = [i for i in range(len(pop))]
    testing_dataset = pd.concat([testing_dataset, pop], axis=0)
    return testing_dataset

