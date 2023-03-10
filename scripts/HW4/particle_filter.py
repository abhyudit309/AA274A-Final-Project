import numpy as np
import scipy.linalg  # You may find scipy.linalg.block_diag useful
import scipy.stats  # You may find scipy.stats.multivariate_normal.pdf useful
from . import turtlebot_model as tb
#import turtlebot_model as tb

EPSILON_OMEGA = 1e-3

class ParticleFilter(object):
    """
    Base class for Monte Carlo localization and FastSLAM.

    Usage:
        pf = ParticleFilter(x0, R)
        while True:
            pf.transition_update(u, dt)
            pf.measurement_update(z, Q)
            localized_state = pf.x
    """

    def __init__(self, x0, R):
        """
        ParticleFilter constructor.

        Inputs:
            x0: np.array[M,3] - initial particle states.
             R: np.array[2,2] - control noise covariance (corresponding to dt = 1 second).
        """
        self.M = x0.shape[0]  # Number of particles
        self.xs = x0  # Particle set [M x 3]
        self.ws = np.repeat(1. / self.M, self.M)  # Particle weights (initialize to uniform) [M]
        self.R = R  # Control noise covariance (corresponding to dt = 1 second) [2 x 2]

    @property
    def x(self):
        """
        Returns the particle with the maximum weight for visualization.

        Output:
            x: np.array[3,] - particle with the maximum weight.
        """
        idx = self.ws == self.ws.max()
        x = np.zeros(self.xs.shape[1:])
        x[:2] = self.xs[idx,:2].mean(axis=0)
        th = self.xs[idx,2]
        x[2] = np.arctan2(np.sin(th).mean(), np.cos(th).mean())
        return x

    def transition_update(self, u, dt):
        """
        Performs the transition update step by updating self.xs.

        Inputs:
            u: np.array[2,] - zero-order hold control input.
            dt: float        - duration of discrete time step.
        Output:
            None - internal belief state (self.xs) should be updated.
        """
        ########## Code starts here ##########
        # TODO: Update self.xs.
        # Hint: Call self.transition_model().
        # Hint: You may find np.random.multivariate_normal useful.
        us = np.random.multivariate_normal(u, self.R, self.xs.shape[0])
        self.xs = self.transition_model(us, dt)
        ########## Code ends here ##########

    def transition_model(self, us, dt):
        """
        Propagates exact (nonlinear) state dynamics.

        Inputs:
            us: np.array[M,2] - zero-order hold control input for each particle.
            dt: float         - duration of discrete time step.
        Output:
            g: np.array[M,3] - result of belief mean for each particle
                               propagated according to the system dynamics with
                               control u for dt seconds.
        """
        raise NotImplementedError("transition_model must be overridden by a subclass of EKF")

    def measurement_update(self, z_raw, Q_raw):
        """
        Updates belief state according to the given measurement.

        Inputs:
            z_raw: np.array[I,2]   - matrix of I rows containing (alpha, r)
                                     for each line extracted from the scanner
                                     data in the scanner frame.
            Q_raw: [np.array[2,2]] - list of I covariance matrices corresponding
                                     to each (alpha, r) row of z_raw.
        Output:
            None - internal belief state (self.x, self.ws) is updated in self.resample().
        """
        raise NotImplementedError("measurement_update must be overridden by a subclass of EKF")

    def resample(self, xs, ws):
        """
        Resamples the particles according to the updated particle weights.

        Inputs:
            xs: np.array[M,3] - matrix of particle states.
            ws: np.array[M,]  - particle weights.

        Output:
            None - internal belief state (self.xs, self.ws) should be updated.
        """
        r = np.random.rand() / self.M

        ########## Code starts here ##########
        # TODO: Update self.xs, self.ws.
        # Note: Assign the weights in self.ws to the corresponding weights in ws
        #       when resampling xs instead of resetting them to a uniform
        #       distribution. This allows us to keep track of the most likely
        #       particle and use it to visualize the robot's pose with self.x.
        # Hint: To maximize speed, try to implement the resampling algorithm
        #       without for loops. You may find np.linspace(), np.cumsum(), and
        #       np.searchsorted() useful. This results in a ~10x speedup.
        M = self.xs.shape[0]

        """ i = 0
        c = ws[0]
        X = []
        idx = []

        u = np.sum(ws)*(r + np.arange(M)/M)  # size M x 1
        sum = np.cumsum(ws) # size M x 1
        for m in range(M):
            i = np.argmax(sum >= u[m])
            X.append(xs[i])
            idx.append(i) 
        X.append(xs[i])
        #idx.append(i)

        self.xs = np.array(X)
        self.ws = ws[idx]"""

        u = np.sum(ws)*(r + np.arange(M)/M)
        sum = np.tile(np.cumsum(ws), (M,1))
        i = (np.argmax((sum - np.tile(u, (M,1)).T) >= 0, axis=1))
        
        self.xs = xs[i]
        self.ws = ws[i]
        ########## Code ends here ##########

    def measurement_model(self, z_raw, Q_raw):
        """
        Converts raw measurements into the relevant Gaussian form (e.g., a
        dimensionality reduction).

        Inputs:
            z_raw: np.array[I,2]   - I lines extracted from scanner data in
                                     rows representing (alpha, r) in the scanner frame.
            Q_raw: [np.array[2,2]] - list of I covariance matrices corresponding
                                     to each (alpha, r) row of z_raw.
        Outputs:
            z: np.array[2I,]   - joint measurement mean.
            Q: np.array[2I,2I] - joint measurement covariance.
        """
        raise NotImplementedError("measurement_model must be overridden by a subclass of EKF")


class MonteCarloLocalization(ParticleFilter):

    def __init__(self, x0, R, map_lines, tf_base_to_camera, g):
        """
        MonteCarloLocalization constructor.

        Inputs:
                       x0: np.array[M,3] - initial particle states.
                        R: np.array[2,2] - control noise covariance (corresponding to dt = 1 second).
                map_lines: np.array[J,2] - J map lines in rows representing (alpha, r).
        tf_base_to_camera: np.array[3,]  - (x, y, theta) transform from the
                                           robot base to camera frame.
                        g: float         - validation gate.
        """
        self.map_lines = map_lines  # Matrix of J map lines with (alpha, r) as rows
        self.tf_base_to_camera = tf_base_to_camera  # (x, y, theta) transform
        self.g = g  # Validation gate
        super(self.__class__, self).__init__(x0, R)

    def transition_model(self, us, dt):
        """
        Unicycle model dynamics.

        Inputs:
            us: np.array[M,2] - zero-order hold control input for each particle.
            dt: float         - duration of discrete time step.
        Output:
            g: np.array[M,3] - result of belief mean for each particle
                               propagated according to the system dynamics with
                               control u for dt seconds.
        """

        ########## Code starts here ##########
        # TODO: Compute g.
        # Hint: We don't need Jacobians for particle filtering.
        # Hint: A simple solution can be using a for loop for each partical
        #       and a call to tb.compute_dynamics
        # Hint: To maximize speed, try to compute the dynamics without looping
        #       over the particles. If you do this, you should implement
        #       vectorized versions of the dynamics computations directly here
        #       (instead of modifying turtlebot_model). This results in a
        #       ~10x speedup.
        # Hint: This faster/better solution does not use loop and does 
        #       not call tb.compute_dynamics. You need to compute the idxs
        #       where abs(om) > EPSILON_OMEGA and the other idxs, then do separate 
        #       updates for them


        ########## Code ends here ##########

        idx_eps_check_true = np.abs(us[:,1]) < EPSILON_OMEGA
        idx_eps_check_false = np.abs(us[:,1]) > EPSILON_OMEGA

        g = np.empty((us.shape[0],3))

        x_new = self.xs[idx_eps_check_true,0] + us[idx_eps_check_true,0] * np.cos(self.xs[idx_eps_check_true,2]) * dt
        y_new = self.xs[idx_eps_check_true,1] + us[idx_eps_check_true,0] * np.sin(self.xs[idx_eps_check_true,2]) * dt
        theta_new = self.xs[idx_eps_check_true,2] + us[idx_eps_check_true,1] * dt

        g[idx_eps_check_true,0] = x_new
        g[idx_eps_check_true,1] = y_new
        g[idx_eps_check_true,2] = theta_new

        x_new = self.xs[idx_eps_check_false,0] + (us[idx_eps_check_false,0] / us[idx_eps_check_false,1]) * (np.sin(self.xs[idx_eps_check_false,2] + us[idx_eps_check_false,1] * dt) - np.sin(self.xs[idx_eps_check_false,2]))
        y_new = self.xs[idx_eps_check_false,1] - (us[idx_eps_check_false,0] / us[idx_eps_check_false,1]) * (np.cos(self.xs[idx_eps_check_false,2] + us[idx_eps_check_false,1] * dt) - np.cos(self.xs[idx_eps_check_false,2]))
        theta_new = self.xs[idx_eps_check_false,2] + us[idx_eps_check_false,1] * dt

        g[idx_eps_check_false,0] = x_new
        g[idx_eps_check_false,1] = y_new
        g[idx_eps_check_false,2] = theta_new

        return g

    def measurement_update(self, z_raw, Q_raw):
        """
        Updates belief state according to the given measurement.

        Inputs:
            z_raw: np.array[I,2]   - matrix of I rows containing (alpha, r)
                                     for each line extracted from the scanner
                                     data in the scanner frame.
            Q_raw: [np.array[2,2]] - list of I covariance matrices corresponding
                                     to each (alpha, r) row of z_raw.
        Output:
            None - internal belief state (self.x, self.ws) is updated in self.resample().
        """
        xs = np.copy(self.xs)
        ws = np.zeros_like(self.ws)

        ########## Code starts here ##########
        # TODO: Compute new particles (xs, ws) with updated measurement weights.
        # Hint: To maximize speed, implement this without looping over the
        #       particles. You may find scipy.stats.multivariate_normal.pdf()
        #       useful.
        # Hint: You'll need to call self.measurement_model()
        z, Q = self.measurement_model(z_raw, Q_raw)
        if z is None:
            # Don't update if measurement is invalid
            # (e.g., no line matches for line-based EKF localization)
            return

        ws = scipy.stats.multivariate_normal.pdf(z, mean=None, cov=Q)
        ########## Code ends here ##########

        self.resample(xs, ws)

    def measurement_model(self, z_raw, Q_raw):
        """
        Assemble one joint measurement and covariance from the individual values
        corresponding to each matched line feature for each particle.

        Inputs:
            z_raw: np.array[I,2]   - I lines extracted from scanner data in
                                     rows representing (alpha, r) in the scanner frame.
            Q_raw: [np.array[2,2]] - list of I covariance matrices corresponding
                                     to each (alpha, r) row of z_raw.
        Outputs:
            z: np.array[M,2I]  - joint measurement mean for M particles.
            Q: np.array[2I,2I] - joint measurement covariance.
        """
        vs = self.compute_innovations(z_raw, np.array(Q_raw))

        ########## Code starts here ##########
        # TODO: Compute Q.
        # Hint: You might find scipy.linalg.block_diag() useful
        Q = scipy.linalg.block_diag(*Q_raw)
        ########## Code ends here ##########

        return vs, Q

    def compute_innovations(self, z_raw, Q_raw):
        """
        Given lines extracted from the scanner data, tries to associate each one
        to the closest map entry measured by Mahalanobis distance.

        Inputs:
            z_raw: np.array[I,2]   - I lines extracted from scanner data in
                                     rows representing (alpha, r) in the scanner frame.
            Q_raw: np.array[I,2,2] - I covariance matrices corresponding
                                     to each (alpha, r) row of z_raw.
        Outputs:
            vs: np.array[M,2I] - M innovation vectors of size 2I
                                 (predicted map measurement - scanner measurement).
        """
        def angle_diff(a, b):
            a = a % (2. * np.pi)
            b = b % (2. * np.pi)
            diff = a - b
            if np.size(diff) == 1:
                if np.abs(a - b) > np.pi:
                    sign = 2. * (diff < 0.) - 1.
                    diff += sign * 2. * np.pi
            else:
                idx = np.abs(diff) > np.pi
                sign = 2. * (diff[idx] < 0.) - 1.
                diff[idx] += sign * 2. * np.pi
            return diff

        hs = self.compute_predicted_measurements()
        ########## Code starts here ##########
        # TODO: Compute vs (with shape [M x I x 2]).
        # Hint: Simple solutions: Using for loop, for each particle, for each 
        #       observed line, find the most likely map entry (the entry with 
        #       least Mahalanobis distance).
        # Hint: To maximize speed, try to eliminate all for loops, or at least
        #       for loops over J. It is possible to solve multiple systems with
        #       np.linalg.solve() and swap arbitrary axes with np.transpose().
        #       Eliminating loops over J results in a ~10x speedup.
        #       Eliminating loops over I results in a ~2x speedup.
        #       Eliminating loops over M results in a ~5x speedup.
        #       Overall, that's 100x!
        # Hint: For the faster solution, you might find np.expand_dims(), 
        #       np.linalg.solve(), np.meshgrid() useful.
        
        I = z_raw.shape[0]  # Number of observed lines
        J = hs.shape[1]  # Number of predicted lines
        M = hs.shape[0] # Number of points

        """ vs = np.empty((M, I, 2))
        for m in range(M):
            for i in range(I):
                dij_min = np.inf
                for j in range(J):
                    vij = np.array([angle_diff(z_raw[i, 0], hs[m, j, 0]), z_raw[i, 1] - hs[m, j, 1]])
                    dij = np.matmul(vij.T, np.matmul(np.linalg.inv(Q_raw[i]), vij))
                    if dij < dij_min:
                        dij_min = dij
                        vs[m][i] = vij """

        vs = np.empty((M, I, 2))
        for m in range(M):
            for i in range(I):
                vi = np.array([angle_diff(z_raw[i, 0], hs[m, :, 0]), z_raw[i, 1] - hs[m, :, 1]])
                di = np.sum(np.multiply(vi, np.matmul(np.linalg.inv(Q_raw[i]), vi)), axis=0)
                vs[m][i] = vi[:,np.argmin(di)]

        """ vs = np.empty((M, I, 2))
        for m in range(M):
            tiled_z_raw_alpha = np.tile(z_raw[:, 0],(J,1))
            tiled_hs_alpha = np.tile(hs[m, :, 0],(I,1))
            tiled_z_raw_r = np.tile(z_raw[:, 1],(J,1))
            tiled_hs_r = np.tile(hs[m, :, 1],(I,1))

            vi = np.array([angle_diff(tiled_z_raw_alpha.T, tiled_hs_alpha), tiled_z_raw_r.T - tiled_hs_r])
            Q = Q_raw.reshape((2*I,2))
            V = vi.reshape((2*I,J))

            di = np.multiply(np.multiply(np.repeat(vi[:,0,:],2,axis=0), np.expand_dims(Q[:,0], axis=1)) + np.multiply(np.repeat(vi[:,1,:],2,axis=0), np.expand_dims(Q[:,1], axis=1)), vi.reshape((2*I,J)))
            di = np.sum(di.reshape((I,2,J)), axis=1)

            print("Argmin Output: ", np.argmin(di, axis=1))
            print("Shape of Argmin Output: ", np.argmin(di, axis=1).shape)

            print("vi @ mins", vi[:,:,np.argmin(di, axis=1)])
            test = vi[:,:,np.argmin(di, axis=1)]
            print("vi @ mins", test[:,np.arange(I),:])

            vs[m] = vi[:,:,np.argmin(di, axis=1)] """

        ########## Code ends here ##########

        # Reshape [M x I x 2] array to [M x 2I]
        return vs.reshape((self.M,-1))  # [M x 2I]

    def compute_predicted_measurements(self):
        """
        Given a single map line in the world frame, outputs the line parameters
        in the scanner frame so it can be associated with the lines extracted
        from the scanner measurements.

        Input:
            None
        Output:
            hs: np.array[M,J,2] - J line parameters in the scanner (camera) frame for M particles.
        """
        ########## Code starts here ##########
        # TODO: Compute hs.
        # Hint: We don't need Jacobians for particle filtering.
        # Hint: Simple solutions: Using for loop, for each particle, for each 
        #       map line, transform to scanner frmae using tb.transform_line_to_scanner_frame()
        #       and tb.normalize_line_parameters()
        # Hint: To maximize speed, try to compute the predicted measurements
        #       without looping over the map lines. You can implement vectorized
        #       versions of turtlebot_model functions directly here. This
        #       results in a ~10x speedup.
        # Hint: For the faster solution, it does not call tb.transform_line_to_scanner_frame()
        #       or tb.normalize_line_parameters(), but reimplement these steps vectorized.


        ########## Code ends here ##########
        hs = np.empty((self.xs.shape[0], self.map_lines.shape[1], self.map_lines.shape[0]))
        alpha = self.map_lines[0,:]
        r = self.map_lines[1,:]

        for i in range(self.xs.shape[0]):

            x_cam = self.xs[i,0] + self.tf_base_to_camera[0] * np.cos(self.xs[i,2]) - self.tf_base_to_camera[1] * np.sin(self.xs[i,2])
            y_cam = self.xs[i,1] + self.tf_base_to_camera[0] * np.sin(self.xs[i,2]) + self.tf_base_to_camera[1] * np.cos(self.xs[i,2])
            th_cam = self.xs[i,2] + self.tf_base_to_camera[2]

            # line parameters in the camera frame
            alpha_in_cam = alpha - th_cam
            r_in_cam = r - x_cam * np.cos(alpha) - y_cam * np.sin(alpha)

            r_pos_idx = r_in_cam < 0
            alpha_in_cam[r_pos_idx] += np.pi
            r_in_cam[r_pos_idx] *= -1.0

            alpha_in_cam = (alpha_in_cam + np.pi) % (2.0*np.pi) - np.pi
            hs[i] = np.array([alpha_in_cam, r_in_cam]).T

        return hs
        ########## Code ends here ##########