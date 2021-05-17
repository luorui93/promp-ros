import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

class ProMP(object):
    def __init__(self, dt, n_samples, n_basis):
        self.dt = dt
        self.n_samples = n_samples
        self.n_basis = n_basis
        self.n_demos = 0
        self.via_points = []

        # Phi_norm: n_samples x n_basis matrix
        self.Phi_norm = self.generate_basis(self.dt, self.n_samples, self.n_basis)

        # self.import_demonstraions()
        # demos: n_samples x n_demos matrix
        self.demos = self.add_demonstration()

        self.time_scale_trajectory()
        # weights: a dictionary storing information for weight matrix
        self.weights = self.learn_weight()

    def generate_basis(self, dt, n_samples, n_basis):
        """
        Generate Gaussian basis 
        """
        # set initial parameters for Gaussian basis
        basis_center = np.linspace(0, dt*n_samples, n_basis)
        sigma = 0.05

        # n_basis x n_samples matrix
        basis_pdf = np.array(list(
            map(lambda x: multivariate_normal.pdf(x=np.linspace(0,1,n_samples), mean=x, cov=sigma**2), basis_center)))

        # n_samples x n_basis matrix
        Phi = basis_pdf.T

        # normalize basis weights
        Phi_norm = Phi / np.sum(Phi, axis=1)[:, np.newaxis]
        
        return Phi_norm


    def learn_weight(self):
        """
        Learn weights by least square regression
        """
        # with multiple demos, now weight "vector" is a n_basis x n_demos matrix
        weights = {}

        # calculate weights using Ordinary Least Square estimator
        w = np.linalg.solve(self.Phi_norm.T@self.Phi_norm, np.dot(self.Phi_norm.T, self.demos))
        
        # The weight value for each dimension is treated as a Gaussian random variable
        # The mean value is calculated across multiple demos instead of different dimensions.
        mean_w = np.mean(w, axis=1)

        # the input matrix to np.cov should follow the rule: each row contains the measurements of one variable
        # each column contains the one measurement of whole set of variable
        # In short, np.cov(M), M is required to be a n_variable x n_demo matrix
        # https://numpy.org/doc/stable/reference/generated/numpy.cov.html
        # return a n_variable x n_variable matrix, the covariance matrix is defaultly calculated as unbiased one
        cov_w = np.cov(w, bias=True)   #20 x 20 covariance matrix, n_basis=20

        #this is equal to sigma_w = np.std(w, axis=1)
        sigma_w = np.sqrt(cov_w.diagonal())

        weights['w'] = w
        weights['mean_w'] = mean_w
        weights['cov_w'] = cov_w
        weights['sigma_w'] = sigma_w

        return weights

    def add_demonstration(self):
        """
        Add single trajectory(s) as demonstration
        """
        t = np.linspace(0, self.dt*self.n_samples, self.n_samples)
        y1 = [x**2 for x in list(t)]
        y2 = [x**2 + 0.01*np.random.normal(1) for x in list(t)]
        y1 = np.array(y1)[:, np.newaxis]
        y2 = np.array(y2)[:, np.newaxis]
        Y = np.hstack((y1, y2))

        self.n_demos += 2
        
        return Y

    def import_demonstration(self):
        """
        Import demonstrations from files
        """
        pass


    def time_scale_trajectory(self):
        """
        Synchornize trajctories with same time scale
        """
        pass

    def add_viapoints(self, viapoint):
        """
        Add viapoints for conditioning 
        Each viapoint has the following field:
        t: phase time of the viapoint
        mean: mean joint position of the viapoint
        cov: covariance matrix of the viapoint
        """
        self.viapoints.append(viapoint)

    def predict(self):
        """
        Predict updated trajectories conditioning on viapoints
        """
        mean = self.weights['mean_w'].copy()
        cov  = self.weights['cov_w'].copy()

        # iterate over all viapoints to 
        # update posterior mean and covariance matrix by conditioning
        for viapoint in self.viapoints:
            phase = viapoint['t']
            v_mean = viapoint['mean']
            v_cov = viapoint['cov']

            # retrieve the weight vector at the same phase
            basis = self.Phi_norm[phase, :][:, np.newaxis]

            # conditioning
            inv = np.linalg.inv(v_cov + basis.T @ cov @ basis)
            K = cov @ basis @ inv
            mean = mean + K @ (v_mean - np.dot(basis.T, mean))
            cov = cov - K @ np.dot(basis.T, cov)

            # we could output new mean and cov every iteration for online planning

        #compute new trajectory
        new_traj = np.dot(self.Phi_norm, mean)
        return new_traj

    ### define properties
    @property
    def num_demos(self):
        return self.n_demos
    
    @property
    def num_basis(self):
        return self.n_basis

    @property
    def weights(self):
        return self.weights
    
    @property
    def dt(self):
        return self.dt