import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.linalg import block_diag

class ProMP(object):
    def __init__(self, dt, n_samples, n_basis, dof):
        self.dt = dt
        self.n_samples = n_samples
        self.dof = dof
        self.n_basis = n_basis
        self.n_demos = 0
        self.viapoints = []

        # Generate Basis
        # Phi_norm: n_samples x n_basis matrix
        self.Phi_norm = self.generate_basis()
        # diagonally concatenate the big PHI with small basis matrices
        self.PHI = np.zeros((dof*n_samples, dof*n_basis))
        for i in range(dof):
            self.PHI[i*n_samples:(i+1)*n_samples, i*n_basis:(i+1)*n_basis] = self.Phi_norm

    def main(self):
        """
        Main function for ProMP
        """
        # demos: n_samples x n_demos matrix
        # self.demos = self.import_demonstraions()
        # self.demos is a list containing all training trajectories
        # each trajectory is in the format of n_samples x n_dof
        self.demos = self.add_demonstration("training/something")

        # synchronize phase of demonstration trajectories
        self.sync_demos, self.alpha = self.sync_trajectory()

        # learn weight using least square regression
        # weights: a dictionary storing information for weight matrix
        self.weights = self.learn_weight()

        # update trajectory based on viapoints
        self.traj = self.predict()

    def generate_basis(self):
        """
        Generate Gaussian basis 
        """
        # set initial parameters for Gaussian basis
        basis_center = np.linspace(0, self.dt*self.n_samples, self.n_basis)
        sigma = 0.05

        # n_basis x n_samples matrix
        basis_unit = np.array(list(
            map(lambda c: multivariate_normal.pdf(x=np.linspace(0,self.dt*(self.n_samples-1),self.n_samples), mean=c, cov=sigma**2), basis_center)))

        # n_samples x n_basis matrix
        Phi = basis_unit.T

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
        w = np.linalg.solve(self.PHI.T@self.PHI, np.dot(self.PHI.T, self.sync_demos))

        # The weight value for each dimension is treated as a Gaussian random variable
        # The mean value is calculated across multiple demos instead of different dimensions.
        # mean_w vector is a n_dof*n_basis x 1 vector: [w1.T, w2.T, ... w_ndof.T].T
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

    def import_demonstration(self, addr):
        """
        Import demonstrations from files
        """
        data = []
        for i in range(self.n_demos):
            traj = np.loadtxt(open(addr + str(i+1) + ".csv"), delimiter=",")
            data.append(traj)

        return data


    def sync_trajectory(self):
        """
        Synchornize trajctories with same time scale
        """
        traj_len = np.array(list(map(lambda traj: len(traj), self.data)))
        ref_len = int(np.mean(traj_len))
        alpha = traj_len / ref_len
        alpha_mean = np.mean(alpha)
        alpha_var  = np.var(alpha)

        # time synchronize the data to contain same number of points for training
        # resample all trajecotries to have same points as the "ideal" trajectory
        # the new trajectory is obtained by interpolating the original data at scaled time point
        sync_data = []
        for i, a in enumerate(alpha):
            resampled_data = np.empty((ref_len, self.dof))
            for j in range(ref_len):
                # the unwarped phase
                z = j * a
                floor_z = int(z)
                if floor_z == self.data[i].shape[0] - 1:
                    scaled_value = self.data[i][floor_z]
                else:
                    scaled_value = self.data[i][floor_z] + (z-floor_z)*(self.data[i][floor_z+1] - self.data[i][floor_z])
                resampled_data[j] = scaled_value
            sync_data.append(resampled_data)
        return sync_data, alpha

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

            # retrieve the weight vector at the same phase in multiple dof we are interested
            # PHI: dof*n_sample x dof*n_basis
            basis = self.PHI[[phase*+i*self.n_samples for i in range(self.dof)], :]

            if (basis.ndim == 1):
                basis = basis[:, np.newaxis]
            else:
                basis = basis.T

            # conditioning
            inv = np.linalg.inv(v_cov + basis.T @ cov @ basis)
            # K: dof*n_basis x dof
            K = cov @ basis @ inv
            mean = mean + K @ (v_mean - np.dot(basis.T, mean))
            cov = cov - K @ np.dot(basis.T, cov)

            # we could output new mean and cov every iteration for online planning

        # delete viapoints that are already conditioned
        self.viapoints.clear()

        # save new mean and cov, compute new trajectory
        self.weights['mean_w'] = mean
        self.weights['cov_w']  = cov
        new_traj = np.dot(self.PHI, mean)
        # reshape new_traj following the defined trajectory format: n_samples x n_dof
        new_traj = new_traj.reshape((self.dof, self.n_samples)).T
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