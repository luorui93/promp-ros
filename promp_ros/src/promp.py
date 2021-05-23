import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.linalg import block_diag

eps = 1e-10

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()        
        t = (te - ts) * 1000
        print (f"{method.__name__} lasts {t}ms")
        return result    
    return timed

class ProMP(object):
    def __init__(self, dt, n_basis, demo_addr, n_demos):
        self.dt = dt
        self.n_basis = n_basis
        self.n_demos = n_demos
        self.n_samples = 0
        self.dof = 0
        self.demo_addr = demo_addr
        self.viapoints = []
        self.obs_sigma = 5e-2
        self.basis_sigma = self.dt * 5

    def main(self):
        """
        Main function for ProMP
        """
        # demos: n_samples x n_demos matrix
        self.demos = self.import_demonstrations()
        # self.demos is a list containing all training trajectories
        # each trajectory is in the format of n_samples x n_dof
        # self.demos = self.add_demonstration()

        # synchronize phase of demonstration trajectories
        self.sync_demos, self.alpha, self.alpha_mean, self.alpha_std = self.sync_trajectory()
        # start diverging
        # print(self.sync_demos[2])
        # obtain necessary information from synced demonstrations
        self.n_samples = self.sync_demos[0].shape[0]
        self.dof = self.sync_demos[0].shape[1]

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
        # print(self.weights['mean'])

        # print(self.PHI.shape)
        # np.savetxt('../training/trained/weight_cov.csv', self.weights['cov'], delimiter=",")
        # np.savetxt('../training/trained/weight_mean.csv', self.weights['mean'], delimiter=",")
        # update mean and cov based on viapoints
        # self.condition_viapoints()

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
        Y = np.empty((self.dof*self.n_samples, self.n_demos))
        for i, traj in enumerate(self.sync_demos):
            y = traj.flatten(order='F')
            Y[:,i] = y

        # convert Y into compatible format for linear regression
        # print(f"Y shape {Y.shape}")
        w = np.linalg.solve(self.PHI.T@self.PHI, np.dot(self.PHI.T, Y))
        # print(f"w shape: {w.shape}")

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

    def import_demonstrations(self):
        """
        Import demonstrations from files
        """
        data = []
        for i in range(0, self.n_demos):
            # human = np.loadtxt(open(self.demo_addr + str(i+1) + ".csv"), delimiter=",")
            # temporal hack to test program
            robot = np.loadtxt(open('../training/plug/file_output_' + str(i+1)), delimiter=",")
            # traj = np.hstack((human, robot))
            data.append(robot)

        return data


    def sync_trajectory(self):
        """
        Synchornize trajctories with same time scale
        """
        dof = self.demos[0].shape[1]
        traj_len = np.array(list(map(lambda traj: len(traj), self.demos)))
        ref_len = int(np.mean(traj_len))
        alpha = traj_len / ref_len
        alpha_mean = np.mean(alpha)
        alpha_std  = np.std(alpha)

        # time synchronize the data to contain same number of points for training
        # resample all trajecotries to have same points as the "ideal" trajectory
        # the new trajectory is obtained by interpolating the original data at scaled time point
        sync_data = []
        for i, a in enumerate(alpha):
            resampled_data = np.empty((ref_len, dof))
            for j in range(ref_len):
                # the unwarped phase
                z = j * a
                floor_z = int(z)
                if floor_z == self.demos[i].shape[0] - 1:
                    scaled_value = self.demos[i][floor_z]
                else:
                    scaled_value = self.demos[i][floor_z] + (z-floor_z)*(self.demos[i][floor_z+1] - self.demos[i][floor_z])
                resampled_data[j] = scaled_value
            sync_data.append(resampled_data)
        return sync_data, alpha, alpha_mean, alpha_std

    def predict(self, obs_traj):
        """
        Predict new trajectory based on observed trajectory
        Each viapoint has the following field:
        t: phase time of the viapoint
        mean: mean joint position of the viapoint
        cov: covariance matrix of the viapoint

        Returns:
        -------
        new_traj_stat
            a tuple which contains the mean and covariance matrix of the joint position on the predicted trajectory
        """
        # add viapoints from traj
        viapoints = []
        sigma = self.obs_sigma
        for i, p in enumerate(obs_traj):
            vp = {}
            vp['t'] = i
            vp['mean'] = p
            vp['cov'] = sigma * np.identity(self.dof)
            viapoints.append(vp)
        
        # predict new trajectory based on given viapoints
        mean, cov = self.condition_viapoints(viapoints)
        self.weights['updated_mean'] = mean
        self.weights['updated_cov']  = cov

        # returned trajectory tuple (mean, cov)
        new_traj_stat = self.compute_trajectory_stat(self.weights['updated_mean'], self.weights['updated_cov'])
        # print(new_traj.shape)
        return new_traj_stat

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
    
    def estimate_phase(self, partial_traj):
        # partial_traj will be in the standard trajectory format
        # sample some alphas from the trained alpha distribution
        # probably should evenly sample
        alpha_samples = np.linspace(self.alpha_mean - 2*self.alpha_std, self.alpha_mean + 2*self.alpha_std, 10)
        # print(f"alpha samples: \n{alpha_samples}")
        # alpha_samples = np.random.normal(self.alpha_mean, self.alpha_std, 20)

        # print(f"observed traj: \n{partial_traj}")
        # compute the priori probability P(alpha)
        print(self.alpha_std)
        prior_alpha = multivariate_normal.pdf(x = alpha_samples, mean = self.alpha_mean, cov=self.alpha_std**2)

        # do an outer product to compute a grid of z=alpha*t
        nobs_samples = partial_traj.shape[0]
        nobs_dof = partial_traj.shape[1]
        ref_t = np.arange(nobs_samples)
        # print(f"ref_t: {ref_t}")
        z_grid = np.dot(alpha_samples[:,np.newaxis] ,ref_t[:,np.newaxis].T)
        # print(f"z_grdi: {z_grid}")

        obs_interp_traj = []

        for z in z_grid:
            # trajectory interpolates with one z for all dof
            single_traj = np.array(list(map(lambda fp: np.interp(z, ref_t, fp), partial_traj.T))).T
            obs_interp_traj.append(single_traj)
        
        # obs_interp_traj is a 3d matrix
        # first dimension is the number of alpha
        # in which each element is a n_samples(time_index) x dof trajectory
        obs_interp_traj = np.array(obs_interp_traj)

        # compute trajectory's trained mean and cov
        mean, cov = self.compute_trajectory_stat(self.weights['mean'], self.weights['cov'])

        # log posteriori
        log_p = np.zeros(alpha_samples.shape)
        for index in range(nobs_samples):
            # retrieve trajectory point at the same time index with all possible alpha
            sample = obs_interp_traj[:, index, :]

            # compute likelihood for all alpha candidates at the same index 
            # however, due to the partial nature of the observance, some dof might be missing
            # so we are calculating the marginal probability of a multivariate gaussian
            m = mean[[index + i*self.n_samples for i in range(nobs_dof)]]
            # retrieve the correct covriance matrix from the concatenated big one
            x, y = np.mgrid[index:cov.shape[0]:self.n_samples, index:cov.shape[0]:self.n_samples]
            c = cov[x, y]
            # c = np.identity(nobs_dof) * self.obs_sigma

            # print(f"alpha*t sample: \n {sample}")
            likelihood = multivariate_normal.pdf(x = sample, mean = m, cov=c)
            posteriori = likelihood * prior_alpha
            posteriori = posteriori / np.sum(posteriori)
            log_p = log_p + np.log(posteriori)
            # print(f"log posteriori: \n{log_p}")
            # print(f"pdf: {result}")
        
        best_alpha = alpha_samples[np.argmax(log_p)]
        # print(f"best alpha: {best_alpha}")
        return best_alpha

        # for interp_y in obs_interp_traj:
            
    
    def reshape_trajectory(self, column_traj):
        """
        Reshape a n_dof*n_sample x 1 column trajectory to standard format trajectory
        """
        matrix_traj = column_traj.reshape((self.dof, self.n_samples)).T
        return matrix_traj

    def compute_trajectory_stat(self, weight_mean, weight_cov):
        """
        Compute distribution statistics of trajecotry point given weights
        """
        # calculate trajectory mean value for each sample time
        new_traj_mean = np.dot(self.PHI, weight_mean)

        # calculate concatenated covariance matrix for the trajcetory
        # refer to estimate_phase on how to retrive the correct covariance matrix
        new_traj_cov  = self.PHI @ weight_cov @ self.PHI.T 
        # np.savetxt('../training/trained/traj_cov.csv', new_traj_cov, delimiter=",")

        return new_traj_mean, new_traj_cov
    
    def padding_trajectory(self):
        """
        Padding partial trajectory to become the correct shape
        """
        pass
    
    def plot(self, plot_error=False):
        figure1 = plt.figure(1, figsize=(5,5))
        ax = figure1.add_subplot(111)
        ax.set_title("All trajectory")
        traj_mean, traj_cov = self.compute_trajectory_stat(self.weights['mean'], self.weights['cov'])

        # reshape trajectory for plotting
        format_traj = self.reshape_trajectory(traj_mean)

        # only plot one joint 
        joint = 0
        plot_traj = format_traj[:, joint]

        t = np.linspace(0,1,self.n_samples)
        ax.plot(t, plot_traj, 'o')
        if plot_error:
            # we only plot std deviation
            std_full = np.sqrt(traj_cov.diagonal())
            std_full = std_full.reshape((self.dof, self.n_samples)).T
            std_joint = std_full[:, joint]

            upper_y = plot_traj + std_joint
            lower_y = plot_traj - std_joint
            ax.fill_between(t, upper_y.flatten(), lower_y.flatten(), alpha=0.4)
            

        # plt.figure(2, figsize=(5,5))
        # plt.title("J2 trajectory")
        # plt.plot(np.linspace(0,1,self.n_samples), traj[:, 1], 'o')
        plt.show()
    
    def plot_trajectory(self, traj, joint, title, plot_error=False):
        """
        Parameters:
        ----------
        traj
            a tuple contains the column trajectory mean and concatenated trajectory
        joint
            the specific joint trajectory to be plotted
        """
        plt.figure(2, figsize=(5,5))
        plt.title(title)

        mean = traj[0]
        cov  = traj[1]
        # reshape trajectory for plotting
        format_traj = self.reshape_trajectory(mean)

        # only plot one joint 
        plot_traj = format_traj[:, joint]
        t = np.linspace(0,1,len(plot_traj))
        plt.plot(t, plot_traj, 'o')

        if plot_error:
            # we only plot std deviation
            std_full = np.sqrt(cov.diagonal())
            std_full = std_full.reshape((self.dof, self.n_samples)).T
            std_joint = std_full[:, joint][0:len(plot_traj)]

            upper_y = plot_traj + std_joint
            lower_y = plot_traj - std_joint
            plt.fill_between(t, upper_y.flatten(), lower_y.flatten(), alpha=0.4)
        plt.show()

    

    ### define properties
    # @property
    # def num_demos(self):
    #     return self.n_demos
    
    # @property
    # def num_basis(self):
    #     return self.n_basis

    # @property
    # def weights(self):
    #     return self.weights
    
    
    
    #     return self.dt


if __name__ == "__main__":
    pmp = ProMP(dt=0.5, n_basis=20, demo_addr='../training/letter/letterAtr', n_demos=8)

    # train model
    pmp.main()
    # pmp.plot(plot_error=False)
    # mean, _ = pmp.compute_trajectory_stat(pmp.weights['mean'], pmp.weights['cov'])
    # traj = pmp.reshape_trajectory(mean)
    # print(f"trained trajectory mean: \n{traj}")
    # print(f"trained cov: {pmp.weights['cov']}")

    #################################################
    # example for predicting new trajctory given partial observation
    test_traj = np.loadtxt(open("../training/plug/file_output_9"), delimiter=",")
    partial_traj = test_traj[0:10,]

    # we need to pad the partial trajectory as the same standard format, fill unknown value with np.nan
    # here we assume the joint 3 and 4 value are unknown
    # nan_traj = np.empty(partial_traj.shape)
    # nan_traj[:] = np.nan
    # padded_traj = np.hstack((partial_traj, nan_traj))

    # return the updated trajectory mean and cov
    # traj_stat = pmp.predict(padded_traj)
    # reformat the obtained column trajectory
    # mat_traj = pmp.reshape_trajectory(traj_stat[0])

    # pmp.plot_trajectory(traj=traj_stat, joint=0, title="Updated J1 Trajectory", plot_error=False)
    # print(f"updated cov: {pmp.weights['updated_cov']}")

    phase = pmp.estimate_phase(partial_traj)

    