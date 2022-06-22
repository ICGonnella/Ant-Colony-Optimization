import numpy as np
from ACO import ant
from scipy.spatial import distance
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
from py2opt_local.py2opt.routefinder import RouteFinder

class Population:
    def __init__(self, num_pop, g, coord, start_, end_, t_constraints, tau_init=1, q_=100,
                 ro_=0.05, sigma_=0.5, tolerance_=10, alpha_=1, beta_=1, ML_=0, p_=0, LS_=False):
        self.m = num_pop  # number of ants
        self.gain_vector = g  # scores of the nodes
        self.n = len(g)  # dimensionality of the problem
        self.coordinates = np.array(coord)  # coordinates of the nodes
        self.dim = coord.shape[1]  # dimensions of the nodes' space
        self.T_max = t_constraints  # constraint on the maximum time allowed
        self.start = start_  # index of the starting node in the "g" vector
        self.end = end_  # index of the ending node in the "g" vector
        self.init_tau = tau_init  # value used to fill the initialized tau matrix
        self.y_best = 0  # initialization of the best objective value found
        self.trajectory_best = [0 for _ in range(self.n)]  # initialization of the best trajectory found
        self.cost_matrix = np.zeros((self.n, self.n))  # initialization of the cost matrix
        self.eta_matrix = np.ones((self.n, self.n))  # initialization of the eta matrix
        self.tau_matrix = np.zeros((self.n, self.n))   # initialization of the tau matrix

        # parameter that discriminates between traditional ACO methods and ML-ACO ones
        self.ML = ML_  # 0-> traditional , 1-> ML-ACO_eta , 2-> ML-ACO_eta^ , 3-> ML-ACO_tau
        self.p = p_  # probability matrix needed for the ML-ACO methods

        # parameter that gives information about wanting or not to perform ACO_LS algorithm
        self.LS_ = LS_

        # parameters needed for both AS and MMAS methods
        self.alpha = alpha_
        self.beta = beta_
        self.Q = q_
        self.ro = ro_

        # extra parameters and variables needed for the MMAS method
        self.sigma = sigma_
        self.tolerance = tolerance_
        self.tau_max = 0
        self.tau_min = 0

        # initialization of the "m" ants
        self.ants = [ant.Ant(self.T_max, self.n, self.alpha, self.beta) for i in range(self.m)]

    def __repr__(self):
        return f"population size = {self.m}, number of possible nodes = {self.n}, T_max = {self.T_max}"

    def reorder(self):
        # put the start node's gain and coordinates at the beginning
        tmp = self.gain_vector[self.start].copy()
        self.gain_vector[self.start] = self.gain_vector[0].copy()
        self.gain_vector[0] = tmp
        tmp = self.coordinates[self.start].copy()
        self.coordinates[self.start] = self.coordinates[0].copy()
        self.coordinates[0] = tmp

        # put the last node's gain and coordinates at the end
        tmp = self.gain_vector[self.end].copy()
        self.gain_vector[self.end] = self.gain_vector[-1].copy()
        self.gain_vector[-1] = tmp
        tmp = self.coordinates[self.end].copy()
        self.coordinates[self.end] = self.coordinates[-1].copy()
        self.coordinates[-1] = tmp

    def cost_matrix_(self):
        for i in range(self.n):
            self.cost_matrix[i] = [distance.euclidean(self.coordinates[i],
                                   self.coordinates[j]) for j in range(self.n)]

    def tau_init(self):
        if self.ML != 3:
            self.tau_matrix = np.full((self.n, self.n), self.init_tau)
        else:
            self.tau_matrix = self.p.copy()

    def eta_matrix_(self):
        self.eta_matrix = np.ones((self.n, self.n))
        if self.ML == 1 or self.ML == 2:
            self.eta_matrix = self.p.copy()
        if self.ML != 1:
            for i in range(self.n):
                for j in range(self.n):
                    if self.cost_matrix[i, j] == 0:
                        self.eta_matrix[i, j] = 0
                    else:
                        self.eta_matrix[i, j] *= self.gain_vector[j]/self.cost_matrix[i, j]

    def init_finalize(self):
        self.reorder()
        self.cost_matrix_()
        self.tau_init()
        self.eta_matrix_()

    def init_ants(self):
        for a in self.ants:
            a.eta_matrix = self.eta_matrix
            a.cost_matrix = self.cost_matrix
            a.gain_vector = self.gain_vector
            a.tau_matrix = self.tau_matrix

    def update_ants(self):
        for a in self.ants:
            a.tau_matrix = self.tau_matrix
            a.clear_trajectory()

    def delta_tau_AS(self, ant_):
        '''
        :param ant_: Ant object, whose delta_tau_matrix we want to update (AS method)
        '''
        ant_.delta_tau_matrix = np.full((self.n, self.n),
                                        ant_.y/(self.Q * self.y_best)) * ant_.X_matrix

    def delta_tau_MMAS(self, ant_):
        '''
        :param ant_: Ant object, whose delta_tau_matrix we want to update (MMAS method)
        '''
        ant_.delta_tau_matrix = np.full((self.n, self.n),
                                        (1 / ant_.y)) * ant_.X_matrix

    def update_tau(self, delta):
        '''
        :param delta: delta_tau for which the tau_matrix is updated
        '''
        self.tau_matrix = (1-self.ro)*self.tau_matrix + delta

    def adjust_tau(self):  # for MMAS method
        for i in range(self.n):
            for j in range(self.n):
                if self.tau_matrix[i, j] > self.tau_max: self.tau_matrix[i, j] = self.tau_max
                if self.tau_matrix[i, j] < self.tau_min: self.tau_matrix[i, j] = self.tau_min

    def pheromone_trail_smoothing(self):  # for MMAS method
        for i in range(self.n):
            for j in range(self.n):
                self.tau_matrix[i, j] = self.tau_matrix[i, j] + self.sigma * (self.tau_max - self.tau_matrix[i, j])

    def LS(self, ant_):
        '''
        :param ant_: the ant object whose trajectory found wants to be improved by a local
        2-opt search
        :return: the ant object with the trajectory and best objective value found updated
        '''
        # nodes not included in the trajectory found
        left_out = [i for i in range(self.n) if not np.isin(ant_.trajectory, i).any()]
        for i in left_out:
            # nodes not included in the trajectory found after a certain number of LS iterations
            left_out_ = [i for i in range(self.n) if not np.isin(ant_.trajectory, i).any()]
            # add the final destination so that it does not change position
            out_ = left_out_ + [self.n - 1]
            # add to the trajectory a new node
            traj_trial = ant_.trajectory[:-1] + [i]
            traj_trial.sort()
            # compute the distance matrix between the nodes included in the new trajectory
            dist_mat = self.cost_matrix.copy()
            dist_mat = np.delete(dist_mat, [k for k in out_ if k != i], axis=0)
            dist_mat = np.delete(dist_mat, [k for k in out_ if k != i], axis=1)
            # 2-opt LS algorithm
            min_cost, best_traj = RouteFinder(dist_mat, traj_trial, iterations=5).solve()
            min_cost += self.cost_matrix[best_traj[-1],self.n - 1]
            best_traj.append(self.n - 1)
            gain = np.sum([self.gain_vector[k] for k in best_traj])
            # substitute the found trajectory to the old one if it brings an improvement
            if min_cost <= self.T_max and gain > ant_.y:
                ant_.trajectory = best_traj.copy()
                ant_.y = gain.copy()
        return ant_

    def AS(self, K):
        '''
        :param K: number of iterations
        '''
        self.tau_init()
        self.eta_matrix_()
        self.init_ants()
        self.y_best = 0
        for k in range(K):
            delta_sum = np.zeros((self.n, self.n))
            for a in self.ants:
                a.create_path()
                if self.LS_:
                    a = self.LS(a)
                if a.y > self.y_best:
                    self.y_best = a.y
                    self.trajectory_best = a.trajectory
                self.delta_tau_AS(a)
                delta_sum = delta_sum + a.delta_tau_matrix
            self.update_tau(delta_sum)
            print(f"{k}, y_max = {self.y_best}, best trajectory: {self.trajectory_best}")
            self.update_ants()

    def MMAS(self, K, global_=False):
        '''
        :param K: number of iterations
        '''
        self.tau_init()
        self.eta_matrix_()
        self.init_ants()
        self.y_best = 0
        delta_tau_best = 0
        stall = 0
        for k in range(K):
            stall += 1
            if stall >= self.tolerance:
                self.pheromone_trail_smoothing()
                stall = 0
            for a in self.ants:
                a.create_path()
                if self.LS_:
                    a = self.LS(a)
                    #return a
                if a.y > self.y_best:
                    stall = 0
                    self.y_best = a.y
                    self.trajectory_best = a.trajectory
                    if global_:
                        self.delta_tau_MMAS(a)
                        delta_tau_best = a.delta_tau_matrix.copy()
                    self.tau_max = 1/(self.ro * self.y_best)
                    self.tau_min = self.tau_max/(2 * self.n)
            if not global_:
                self.delta_tau_MMAS(self.ants[np.argmax([self.ants[i].y for i in range(self.m)])])
                delta_tau_best = self.ants[np.argmax([self.ants[i].y for i
                                                      in range(self.m)])].delta_tau_matrix.copy()
            self.update_tau(delta_tau_best)
            self.adjust_tau()
            print(
                f"{k}, y_max = {self.y_best}, y_max found = {np.max([self.ants[i].y for i in range(self.m)])},"
                f" best trajectory: {self.trajectory_best}")
            self.update_ants()

    def plot(self):
        assert self.dim == 2, "the plot function is designed for problems in 2 dimensions"
        fig, ax = plt.subplots(1, 1, figsize=(20,5) )
        ax.plot(self.coordinates[:, 0], self.coordinates[:, 1], 'o', markersize=2)
        ax.plot(self.coordinates[0, 0], self.coordinates[0, 1], 'o', markersize=6, color="red")
        ax.plot(self.coordinates[self.n-1, 0], self.coordinates[self.n-1, 1], 'o', markersize=6, color="red")
        i = 1
        for d_x, d_y in zip(self.coordinates[:, 0], self.coordinates[:, 1]):
            if d_x == self.coordinates[0, 0] and d_y == self.coordinates[0, 1]:
                ax.text(d_x, d_y, str(i) + " (START)", color="black", fontsize=10)
            else:
                if d_x == self.coordinates[self.n -1, 0] and d_y == self.coordinates[self.n -1, 1]:
                    ax.text(d_x, d_y, str(i) + " (END)", color="black", fontsize=10)
                else:
                    ax.text(d_x, d_y, str(i), color="black", fontsize=10)
            i += 1
        ax.set_xticks([5 * i for i in range(21)])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        for i in range(1, len(self.trajectory_best)):
            ax.plot([self.coordinates[self.trajectory_best[i-1], 0], self.coordinates[self.trajectory_best[i], 0]],
                    [self.coordinates[self.trajectory_best[i-1], 1], self.coordinates[self.trajectory_best[i], 1]])
        ax.set_title(f"Best solution found, objective function value equal to {self.y_best}")
        plt.show()

    def f1_(self):
        f1 = np.zeros((self.n, self.n))
        for i in range(self.n - 1):
            for j in range(1, self.n):
                if j != i:
                    f1[i, j] = self.cost_matrix[i, j] / self.T_max
        return f1

    def ratio_matrix(self):
        tmp = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if self.cost_matrix[i, j] < 1e-10:
                    tmp[i, j] = 0
                else:
                    tmp[i, j] = self.gain_vector[j] / self.cost_matrix[i, j]
        return tmp

    def f2_(self):
        f2 = np.zeros((self.n, self.n))
        tmp = self.ratio_matrix()
        for i in range(self.n - 1):
            for j in range(1, self.n):
                if j != i:
                    f2[i, j] = tmp[i, j] / np.max(tmp[i])
        return f2

    def f3_(self):
        f3 = np.zeros((self.n, self.n))
        tmp = self.ratio_matrix()
        for i in range(self.n - 1):
            for j in range(1, self.n):
                if j != i:
                    if np.max(tmp[:, j]) < 1e-15:
                        f3[i, j] = 0
                    else:
                        f3[i, j] = tmp[i, j] / np.max(tmp[:, j])
        return f3

    def random_sampling(self):
        trajectory = []
        y = []
        for k in range(self.n * 100):
            t = 0
            v_c = 0
            traj_c = [0]
            y_c = 0
            perm = np.random.permutation([i for i in range(1, self.n - 1)])
            for i in perm:
                if t + self.cost_matrix[v_c, i] + self.cost_matrix[i, self.n - 1] <= self.T_max:
                    traj_c.append(i)
                    t += self.cost_matrix[v_c, i]
                    y_c += self.gain_vector[i]
                    v_c = i
            trajectory.append(traj_c)
            y.append(y_c)
        return pd.DataFrame({"t": trajectory, "g": y})

    def fr_fc(self):
        tmp = np.array(self.random_sampling().sort_values("g", ascending=False))
        mean_y = np.mean(tmp[:, 1])
        diff_y = np.sum([(k - mean_y) for k in tmp[:, 1]])
        var_y = np.sum([(k - mean_y)**2 for k in tmp[:, 1]])

        f_r = np.zeros((self.n, self.n))
        mean_X = np.zeros((self.n, self.n))
        S_1 = np.zeros((self.n, self.n))

        for k in range(len(tmp)):
            for i in range(1, len(tmp[k, 0])):
                f_r[tmp[k, 0][i-1], tmp[k, 0][i]] += 1/(k+1)
                mean_X[tmp[k, 0][i-1], tmp[k, 0][i]] += 1/len(tmp)
                S_1[tmp[k, 0][i-1], tmp[k, 0][i]] += tmp[k, 1] - mean_y

        f_c = np.zeros((self.n, self.n))
        for i in range(self.n - 1):
            for j in range(1, self.n):
                if j != i:
                    var_c = (1 - mean_X[i, j]) * S_1[i, j] - mean_X[i, j] * (diff_y - S_1[i, j])
                    var_x = mean_X[i, j] * (1 - mean_X[i, j]) * len(tmp)
                    if np.abs(mean_X[i, j]) <= 1e-15:
                        f_c[i, j] = 0
                    if np.abs(mean_X[i, j] - 1) <= 1e-15:
                        f_c[i, j] = 1
                    if np.abs(mean_X[i, j]) > 1e-15 and np.abs(mean_X[i, j] - 1) > 1e-15:
                        f_c[i, j] = var_c / np.sqrt(var_x * var_y)
        return f_r, f_c

    def f4_f5(self):
        fr, fc = self.fr_fc()
        return fr / np.max(fr), fc / np.max(fc)

    def features(self):
        f1 = self.f1_()
        f2 = self.f2_()
        f3 = self.f3_()
        f4, f5 = self.f4_f5()
        features_ = pd.DataFrame()
        for i in range(self.n * self.n):
            if i // self.n != i % self.n and i // self.n != self.n - 1 and i % self.n != 0 and i % self.n != self.n - 1:
                features_ = pd.concat(
                    [features_, pd.DataFrame({"f1": [f1[i // self.n, i % self.n]],
                                             "f2": [f2[i // self.n, i % self.n]],
                                             "f3": [f3[i // self.n, i % self.n]],
                                             "f4": [f4[i // self.n, i % self.n]],
                                             "f5": [f5[i // self.n, i % self.n]]})])
        return features_
