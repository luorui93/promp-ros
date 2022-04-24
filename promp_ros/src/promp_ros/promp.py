import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.linalg import block_diag

import rospy
from rospy.core import rospyerr


class ProMP(object):
    def __init__(self, dt, n_basis, demo_addr, n_demos, n_dof=7):
        self.dt = dt
        self.n_basis = n_basis
        self.n_demos = n_demos
        self.n_samples = 0
        self.dof = n_dof 
        self.demo_addr = demo_addr
        self.viapoints = []
        self.obs_sigma = 1e-3
        self.basis_sigma = self.dt * 5
        self.fix_length = True

    def main(self):
        """
        Main function for ProMP
        """
        # demos: n_samples x n_demos matrix
        cols = [7, 8, 9, 10, 11, 12, 13, 21, 22, 23]
        self.demos = self.import_demonstrations(cols)
        self.sync_demos = self.demos
        self.ref_len = self.demos[0].shape[0]
        self.alpha_mean = 1.0

        # obtain necessary information from synced demonstrations
        self.n_samples = self.sync_demos[0].shape[0]


        # Generate Basis
        # Phi_norm: n_samples x n_basis matrix
        self.Phi_norm = self.generate_basis()
        
        # diagonally concatenate the big PHI with basis matrices
        self.PHI = np.zeros((self.dof*self.n_samples, self.dof*self.n_basis))
        for i in range(self.dof):
            self.PHI[i*self.n_samples:(i+1)*self.n_samples, i*self.n_basis:(i+1)*self.n_basis] = self.Phi_norm

        # learn weight using least square regression
        # weights: a dictionary storing information for weight matrix
        self.weights = self.learn_weight()


    def import_demonstrations(self, cols=None):
        """
        Import demonstrations from files
        """
        data = []
        ref_len = 0
        for i in range(0, self.n_demos):
            # human = np.loadtxt(open(self.demo_addr + str(i+1) + ".csv"), delimiter=",")
            # temporal hack to test program
            try:
                addr = self.demo_addr + str(i+1) + ".csv"
                training_data = np.loadtxt(open(addr), delimiter=",")
            except OSError:
                rospy.logwarn(f"{addr} is not found")
                continue
            if (ref_len == 0):
                ref_len = training_data.shape[0]
            if cols is not None:
                if self.fix_length and training_data.shape[0] != ref_len:
                    raise ValueError(f"File {i+1} has data length{training_data.shape[0]}")
                data.append(training_data[:,cols])
            else:
                data.append(training_data)

        return data

    def generate_basis(self, time_sample=None):
        """
        Generate Gaussian basis for training or at given time_sample
        """
        # set initial parameters for Gaussian basis
        duration = self.dt * self.n_samples
        basis_center = np.linspace(0, duration, self.n_basis)
        # SUPER IMPORTANT PARAMETER!!!! Would affect the result a lot!
        cov = self.basis_sigma

        # evenly sample gaussian basis
        if time_sample is None:
            time_sample = np.linspace(0, duration-self.dt, self.n_samples)
        # n_basis x n_samples matrix
        basis_unit = np.array(list(
            # evenly sample n_samples gaussian bases 
            map(lambda c: multivariate_normal.pdf(x=time_sample, mean=c, cov=cov), basis_center)))

        # n_samples x n_basis matrix
        Phi = basis_unit.T

        # normalize basis weights
        Phi_norm = Phi / np.sum(Phi, axis=1)[:, np.newaxis]
        # print(time_sample)
        
        return Phi_norm
    

    def learn_weight(self):
        """
        Learn weights by least square regression
        """
        # with multiple demos, now weight "vector" is a n_basis x n_demos matrix
        weights = {}

        # calculate weights using Ordinary Least Square estimator
        # the output weight vector should be the format:
        # [w1.T, w2.T, w3.T ... w_ndof.T].T
        # the input PHI matrix is a diagnoal matrix comprised of Phi_norm
        # the dependent vector (y) should be the format:
        # [y11, y21, y31,... y_sample_1, y12, y22, y32, ... y_sample_2, ... y_sample_ndof].T
        # the vector is listed in sample-first order
        Y = np.empty((self.dof*self.n_samples, len(self.sync_demos)))
        for i, traj in enumerate(self.sync_demos):
            y = traj.flatten(order='F')
            Y[:,i] = y

        # convert Y into compatible format for linear regression
        # print(f"Y shape {Y.shape}")
        w = np.linalg.solve(self.PHI.T@self.PHI, np.dot(self.PHI.T, Y))

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

        weights['w'] = w
        weights['mean'] = mean_w
        weights['cov'] = cov_w
        # print(f"weights first row: {w[0,:]}")
        # print(f"weight cov (0,0): {cov_w[0,0]}")
        # print(f"weights: {weights['w']}")

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


    def predict(self, obs_traj, phase_estimation=True, given_time=None):
        """
        Predict new trajectory based on observed trajectory
        Each viapoint has the following field:
        t: phase time of the viapoint
        mean: mean joint position of the viapoint
        cov: covariance matrix of the viapoint
        given_time: specify the time for each point in obs_traj

        Returns:
        -------
        new_traj_stat
            a tuple which contains the mean and covariance matrix of the joint position on the predicted trajectory
        """
        # estimate alpha and resample the observation trajectory accordingly
        phase = obs_traj.shape[0] / self.ref_len
        
        # add viapoints from traj
        viapoints = []
        sigma = self.obs_sigma
        if given_time is None:
            for i, p in enumerate(obs_traj):
                vp = {}
                vp['t'] = i
                vp['mean'] = p
                vp['cov'] = sigma * np.identity(self.dof)
                viapoints.append(vp)
        else:
            assert(len(given_time) == len(obs_traj)), "The given time should be the same length as obs_traj"
            for i in range(len(given_time)):
                rospy.loginfo("IN TEST")
                vp = {}
                vp['t'] = given_time[i]
                vp['mean'] = obs_traj[i]
                vp['cov'] = sigma * np.identity(self.dof)
                viapoints.append(vp)
        
        # predict new trajectory based on given viapoints
        mean, cov = self.condition_viapoints(viapoints)
        self.weights['updated_mean'] = mean
        self.weights['updated_cov']  = cov

        # returned trajectory tuple (mean, cov)
        new_traj_stat = self.compute_trajectory_stat(self.weights['updated_mean'], self.weights['updated_cov'])
        # print(new_traj.shape)
        return new_traj_stat, phase

    def condition_viapoints(self, viapoints):
        """
        Update posterior mean and cov by conditioning on viapoints/observations
        Each viapoint contains the joint position mean/cov and synchronized timestep

        Returns:
        -------
        mean
            the updated weight vector means
        cov 
            the updated weight vector covrariance
        """
        mean = self.weights['mean'].copy()
        cov  = self.weights['cov'].copy()
        # print(f"cov: {cov}")

        # iterate over all viapoints to 
        # update posterior mean and covariance matrix by conditioning
        for viapoint in viapoints:
            phase = viapoint['t']
            v_mean = viapoint['mean']
            v_cov = viapoint['cov']

            # retrieve valid dof index
            valid_dof = np.arange(0, self.dof)
            valid_dof = valid_dof[~np.isnan(v_mean)]
            # non valid basis will stay zero
            # PHI: dof*n_sample x dof*n_basis
            # retrieve the basis at the same phase and valid dof
            basis = np.zeros((self.dof, self.dof*self.n_basis))
            basis[valid_dof,:] = self.PHI[[phase+i*self.n_samples for i in valid_dof],:]
            # print(basis)

            if (basis.ndim == 1):
                basis = basis[:, np.newaxis]
            else:
                basis = basis.T
            
            # now set all NaN to 0
            v_mean[np.isnan(v_mean)] = 0.0

            # conditioning
            inv = np.linalg.inv(v_cov + basis.T @ cov @ basis)
            # K: dof*n_basis x dof
            K = cov @ basis @ inv
            mean = mean + K @ (v_mean - np.dot(basis.T, mean))
            cov = cov - K @ np.dot(basis.T, cov)

            # we could output new mean and cov every iteration for online planning

        # delete viapoints that are already conditioned
        # self.viapoints.clear()

        # save new mean and cov, compute new trajectory
        return mean, cov
    

    def compute_trajectory_stat(self, weight_mean, weight_cov):
        """
        Compute distribution statistics of trajecotry point given weights
        """
        # calculate trajectory mean value for each sample time
        new_traj_mean = np.dot(self.PHI, weight_mean)

        # calculate concatenated covariance matrix for the trajcetory
        # refer to estimate_phase on how to retrive the correct covariance matrix
        new_traj_cov  = self.PHI @ weight_cov @ self.PHI.T 
        # np.savetxt(self.demo_addr+'traj_cov.csv', new_traj_cov, delimiter=",")

        return new_traj_mean, new_traj_cov
    