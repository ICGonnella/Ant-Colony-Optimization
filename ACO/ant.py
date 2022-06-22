import numpy as np


class Ant:
    def __init__(self, t_constraint, n, alpha=1, beta=1):
        self.T_max = t_constraint  # constraint on the maximum time allowed
        self.n = n  # dimensionality of the problem
        self.t = 0  # starting time is 0
        self.trajectory = [0]  # starting node is 0
        self.y = 0  # starting trajectory gain is 0

        # parameters needed for ACO methods
        self.alpha = alpha
        self.beta = beta

        # matrices needed for ACO method
        self._tau_matrix = []
        self._cost_matrix = []
        self._eta_matrix = []
        self.X_matrix = np.zeros((self.n, self.n))
        self._delta_tau_matrix = []
        self._gain_vector = []

    def __repr__(self):
        return f"(T_max = {self.T_max}, n = {self.n}, path = {self.trajectory}, gain = {self.y})"

    @property
    def tau_matrix(self):
        return self._tau_matrix

    @tau_matrix.setter
    def tau_matrix(self, tau):
        assert np.array(tau).shape == (self.n, self.n), \
            f"tau_matrix should be a 2d matrix of shape ({self.n},{self.n}), " \
            f"instead it has shape {np.array(tau).shape}"
        self._tau_matrix = tau

    @property
    def delta_tau_matrix(self):
        return self._delta_tau_matrix

    @delta_tau_matrix.setter
    def delta_tau_matrix(self, delta_tau):
        assert np.array(delta_tau).shape == (self.n, self.n), \
            f"delta_tau_matrix should be a 2d matrix of shape ({self.n},{self.n}), " \
            f"instead it has shape {np.array(delta_tau).shape}"
        self._delta_tau_matrix = delta_tau

    @property
    def eta_matrix(self):
        return self._eta_matrix

    @eta_matrix.setter
    def eta_matrix(self, eta):
        assert np.array(eta).shape == (self.n, self.n), \
            f"eta_matrix should be a 2d matrix of shape ({self.n},{self.n}), " \
            f"instead it has shape {np.array(eta).shape}"
        self._eta_matrix = eta

    @property
    def cost_matrix(self):
        return self._cost_matrix

    @cost_matrix.setter
    def cost_matrix(self, c):
        assert np.array(c).shape == (self.n, self.n), \
            f"cost_matrix should be a 2d matrix of shape ({self.n},{self.n}), " \
            f"instead it has shape {np.array(c).shape}"
        self._cost_matrix = c

    @property
    def gain_vector(self):
        return self._gain_vector

    @gain_vector.setter
    def gain_vector(self, g):
        assert np.array(g).shape == (self.n,), \
            f"gain_vector should be a 1d vector of shape ({self.n}), " \
            f"instead it has shape {np.array(g).shape}"
        self._gain_vector = g

    def trajectory_gain(self):
        self.y = np.sum([self._gain_vector[i] for i in self.trajectory])

    def next_nodes(self):
        # if the T_max constraint is respected
        possible_nodes = [i for i in range(self.n)
                          if (self._cost_matrix[self.trajectory[-1], i]+self._cost_matrix[i, -1]
                              <= (self.T_max-self.t))]
        # if the node is not yet in the trajectory
        tmp = [i for i in possible_nodes if not np.isin(self.trajectory, i).any()]
        return tmp

    def probability_next_nodes(self):
        prob = np.zeros(self.n)
        tmp = self.next_nodes()
        if tmp == [self.n - 1]:
            prob[-1] = 1
        else:
            for i in tmp: prob[i] = (self._tau_matrix[self.trajectory[-1], i] **
                                     self.alpha) * (self._eta_matrix[self.trajectory[-1],
                                                                    i] ** self.beta)
        tmp = np.array(prob)/np.sum(prob)
        return tmp

    def create_path(self):
        while self.trajectory[-1] != (self.n - 1):
            self.trajectory.append(np.random.choice([i for i in range(self.n)], p=self.probability_next_nodes()))
            self.t += self._cost_matrix[self.trajectory[-2], self.trajectory[-1]]
            self.X_matrix[self.trajectory[-2], self.trajectory[-1]] = 1
        self.trajectory_gain()

    def clear_trajectory(self):
        self.trajectory = [0]
        self.t = 0
        self.X_matrix = np.zeros((self.n, self.n))
