import numpy as np
import pandas as pd
import os
from ACO import population


def generate_dataset(n, dim=2):
    '''
    :param n: number of nodes to be generated
    :param dim: number of coordinates for each node
    :return: 2d array of coordinates of n nodes, 1d array of scores associated to each node, T_max
    '''
    coordinates = [[] for _ in range(dim)]
    score = []
    for i in range(n):
        for j in range(dim):
            coordinates[j].append(np.random.uniform(0, 100))
        if i == 0 or i == (n-1):
            score.append(0.)
        else:
            score.append(int(np.random.uniform(0, 100)))
    return np.array(coordinates), np.array(score), int(np.random.uniform(100, 400))


def load_dataset(num, T=True, n_nodes=50):
    '''
    :param num: index of the dataset to be loaded
    :param T: whether or not the dataset should be loaded also with the T_max and start information
    :return: the dataset required as pd.DataFrame
    '''
    assert os.path.exists(
        f"./MLACO/Datasets/random{n_nodes}/0{num}.op") or os.path.exists(
        f"./MLACO/Datasets/random{n_nodes}/00{num}.op"), f"the file number {num} does not exist"
    if num < 10:
        if T:
            d = pd.read_csv(f"./MLACO/Datasets/random{n_nodes}/00{num}.op", sep=" ", header=0)
        else:
            d = pd.read_csv(f"./MLACO/Datasets/random{n_nodes}/00{num}.op", sep=" ", header=0, names=["x",
                                                                                               "y", "score"])
    else:
        if T:
            d = pd.read_csv(f"./MLACO/Datasets/random{n_nodes}/0{num}.op", sep=" ", header=0)
        else:
            d = pd.read_csv(f"./MLACO/Datasets/random{n_nodes}/0{num}.op", sep=" ", header=0, names=["x",
                                                                                               "y", "score"])
    return d


def convert_double_x_coordinate(d):
    '''
    :param d: dataset on which it has to be performed the conversion
    :return: the dataset with the "x" column converted in a double
    '''
    new_x = []
    for i in np.array(d["x"]):
        new_x.append(float(i))
    d["x"] = new_x
    return d


def find_end(d, start):
    '''
    :param d: dataset we want to analyse
    :param start: start point
    :return: end point (the other point that has gain=0)
    '''
    for i in range(len(d)):
        if d["score"][i] == 0 and i != start:
            return i


def load_problem(num, n_nodes=50):
    '''
    :param num: index of the dataset to be loaded
    :return: T_max constraint, start and end nodes' indices and the dataset (as pd.DataFrame)
    '''
    d = load_dataset(num, n_nodes=n_nodes)
    T_max = int(d.columns[0])
    start = int(d.columns[1]) - 1
    d = convert_double_x_coordinate(load_dataset(num, T=False, n_nodes=n_nodes)[:-1])
    end = find_end(d, start)
    return T_max, start, end, d


def load_trajectory(num):
    '''
    :param num: index of the trajectory to be loaded
    :return: loaded trajectory
    '''
    assert os.path.exists(
        f"./MLACO/Datasets/random50/0{num}.opt.tour") or os.path.exists(
        f"./MLACO/Datasets/random50/00{num}.opt.tour"), f"the file number {num} does not exist"
    if num < 10:
        d = pd.read_csv(f"./MLACO/Datasets/random50/00{num}.opt.tour", sep=" ", header=0)
    else:
        d = pd.read_csv(f"./MLACO/Datasets/random50/0{num}.opt.tour", sep=" ", header=0)
    return d


def update_nodes_indices(traj, start, end, n):
    '''
    :param traj: trajectory of interest
    :param start: index for the starting node
    :param end: index of the ending node
    :return: trajectory modified so that the starting node's name is 0 and the end node's name is n-1
    '''
    ind_start, ind_0, ind_end, ind_n = -1, -1, -1, -1
    if np.isin(np.array(traj), start).any(): ind_start = np.where(np.array(traj) == start)[0][0]
    if np.isin(np.array(traj), 0).any(): ind_0 = np.where(np.array(traj) == 0)[0][0]
    if np.isin(np.array(traj), end).any(): ind_end = np.where(np.array(traj) == end)[0][0]
    if np.isin(np.array(traj), n-1).any(): ind_n = np.where(np.array(traj) == n-1)[0][0]

    if ind_start != -1: traj[ind_start] = 0
    if ind_0 != -1: traj[ind_0] = start
    if ind_end != -1: traj[ind_end] = n-1
    if ind_n != -1: traj[ind_n] = end

    return traj


def convert_integer_trajectory(traj):
    for i in range(len(traj)):
        traj[i] = int(traj[i])
    return traj


def load_solution(num, start, end, n):
    '''
    :param num: index of the solution to be loaded
    :param start: index of the starting node
    :param end: index of the ending node
    :return: trajectory updated with the correct nodes' names
    '''
    trajectory = convert_integer_trajectory(np.array(load_trajectory(num)[:-1])).reshape(-1)
    trajectory = [i-1 for i in trajectory]
    return update_nodes_indices(trajectory, start, end, n)


def update_coordinates_indices(d, start, end, n):
    tmp = d.iloc[0, :]
    d.iloc[0, :] = d.iloc[start, :]
    d.iloc[start, :] = tmp

    tmp = d.iloc[n-1, :]
    d.iloc[n-1, :] = d.iloc[end, :]
    d.iloc[end, :] = tmp
    return d


def plot_solution(num, start, end, n):
    traj = load_solution(num, start, end, n)
    T_max, start, end, d = load_problem(num)
    d = update_coordinates_indices(d, start, end, n)
    tmp = population.Population(1, np.array(d["score"]), np.array(d.iloc[:, :-1]), start, end, T_max, 0.1, 100, 0.05, 0.5)
    tmp.trajectory_best = traj
    tmp.y_best = np.sum([tmp.gain_vector[i] for i in tmp.trajectory_best])
    tmp.plot()
lala = np.ones(10)


def labels(traj, n):
    lab = np.full((n, n), "neg")
    for i in range(len(traj)): lab[traj[i-1], traj[i]] = "pos"
    lab = lab[:-1, 1:-1].reshape(-1)
    lab = np.array([lab[i] for i in range((n - 1) * (n - 2)) if i // (n-2) - i % (n-2) != 1])
    return lab


def from_x_to_trajectory(x):
    traj = [0]
    while traj[-1] != len(x) - 1:
        traj.append(np.where(x[traj[-1], :] == 1)[0][0])
    return traj


