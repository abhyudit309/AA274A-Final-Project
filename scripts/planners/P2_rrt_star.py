import numpy as np

class RRTStar(object):
    """ Represents a motion planning problem to be solved using the RRT* algorithm"""

    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, occupancy):
        self.statespace_lo = np.array(statespace_lo)  # state space lower bound (e.g., [-5, -5])
        self.statespace_hi = np.array(statespace_hi)  # state space upper bound (e.g., [5, 5])
        self.x_init = np.array(x_init)  # initial state
        self.x_goal = np.array(x_goal)  # goal state
        self.occupancy = occupancy  # occupancy grid (a StochOccupancyGrid2D object)
        self.path = None  # the final path as a list of states

    def is_free_motion(self, x1, x2):
        """
        Subject to the robot dynamics, returns whether a point robot moving
        along the shortest path from x1 to x2 would collide with any obstacles
        (implemented as a "black box")

        Inputs:
            x1: start state of motion
            x2: end state of motion
        Output:
            Boolean True/False
        """
        if not self.occupancy.is_free(x2):
            return False

        distance = np.linalg.norm(x2 - x1)
        if distance <= 0.005:
            return True

        N = int(distance/(0.15 / 50))
        step = np.linspace(0, distance, num=N)
        step = step[1:N - 1]
        free = []

        for i in range(len(step)):
            state = x1 + step[i] * (x2 - x1) / distance
            free.append(self.occupancy.is_free(state))

        return all(free)

    def find_nearest(self, V, x):
        """
        Given a list of states V and a query state x, returns the index (row)
        of V such that the steering distance (subject to robot dynamics) from
        V[i] to x is minimized

        Inputs:
            V: np.array of states ("samples")
            x: query state
        Output:
            Integer index of the nearest point in V to x
        """
        return np.argmin([np.linalg.norm(x - state) for state in V])

    def steer_towards(self, x1, x2, eps):
        """
        Steers from x1 towards x2 along the shortest path (subject to robot
        dynamics). Returns x2 if the length of this shortest path is less than
        eps, otherwise returns the point at distance eps along the path from
        x1 to x2.

        Inputs:
            x1: start state
            x2: target state
            eps: maximum steering distance
        Output:
            State (numpy vector) resulting from bounded steering
        """
        steering_distance = np.linalg.norm(x2 - x1)
        if steering_distance < eps:
            return x2
        else:
            return x1 + eps * (x2 - x1) / steering_distance

    def near(self, V, x, r):
        """
        Finds all the neighbors of x in V that are within radius r.

        Inputs:
            V: np.array of states ("samples")
            x: target state
            r: radius
        Output:
            State (numpy vector) resulting from bounded steering
        """
        dist = [np.linalg.norm(x - state) for state in V]
        idx = [i for i in range(len(dist)) if dist[i] < r]
        return V[idx]

    def distance(self, x1, x2):
        """
        Computes the Euclidean distance between two states.

        Inputs:
            x1: First state
            x2: Second state
        Output:
            Float Euclidean distance
        """
        return np.linalg.norm(x1 - x2)

    def parent(self, V, P, x):
        """
        Finds the parent of the input state x.

        Inputs:
            V: np.array of states ("samples")
            P: np.array of parent indices
            x: target state
        Output:
            State in V that is parent of x
        """
        idx = np.where((V == x).all(axis=1))[0][0]
        return V[P[idx]]

    def cost(self, V, P, x):
        """
        Finds the cost of the path from x_init to input state x.

        Inputs:
            V: np.array of states ("samples")
            P: np.array of parent indices
            x: target state
        Output:
            Float cost
        """
        Px = self.parent(V, P, x)
        dist = self.distance(Px, x)
        while (Px != self.x_init).any():
            x = Px
            Px = self.parent(V, P, x)
            dist = dist + self.distance(Px, x)
        return dist

    def solve(self, eps=0.5, max_iters=3000, gamma=10, goal_bias=0.05):
        """
        Constructs an RRT* rooted at self.x_init with the aim of producing a
        dynamically-feasible and obstacle-free trajectory from self.x_init
        to self.x_goal.

        Inputs:
            eps: maximum steering distance
            max_iters: maximum number of RRT* iterations (early termination
                is possible when a feasible solution is found)
            goal_bias: probability during each iteration of setting
                x_rand = self.x_goal (instead of uniformly randomly sampling
                from the state space)
        Output:
            bool, whether path was found or not (and plots).
        """

        state_dim = len(self.x_init)

        # V stores the states that have been added to the RRT* (pre-allocated at its maximum size
        # since numpy doesn't play that well with appending/extending)
        V = np.zeros((max_iters + 1, state_dim))
        V[0, :] = self.x_init  # RRT* is rooted at self.x_init
        n = 1  # the current size of the RRT* (states accessible as V[range(n),:])

        # P stores the parent of each state in the RRT. P[0] = -1 since the root has no parent,
        # P[1] = 0 since the parent of the first additional state added to the RRT must have been
        # extended from the root.
        P = -np.ones(max_iters + 1, dtype=int)

        success = False

        # Main algorithm
        for k in range(max_iters):
            z = np.random.uniform(0, 1)
            if z < goal_bias:
                x_rand = self.x_goal
            else:
                x_rand = np.array([np.random.uniform(self.statespace_lo[i], self.statespace_hi[i])
                                   for i in range(state_dim)])
            nearest_state_index = self.find_nearest(V[range(n), :], x_rand)
            x_nearest = V[nearest_state_index, :]
            x_new = self.steer_towards(x_nearest, x_rand, eps)
            if self.is_free_motion(x_nearest, x_new):
                V[n, :] = x_new
                x_min = x_nearest
                r = min((gamma / np.pi * np.log(n) / n) ** (1 / state_dim), eps)
                X_near = self.near(V[range(n), :], x_new, r)
                cost_x_new = np.inf
                for x_near in X_near:
                    if self.is_free_motion(x_near, x_new):
                        c = self.cost(V[range(n), :], P[range(n)], x_near) + self.distance(x_near, x_new)
                        if c < cost_x_new:
                            cost_x_new = c
                            x_min = x_near
                P[n] = np.where((V[range(n), :] == x_min).all(axis=1))[0][0]
                for x_near in X_near:
                    if (x_near != x_min).any():
                        if (self.is_free_motion(x_new, x_near)) and (
                                self.cost(V[range(n + 1), :], P[range(n + 1)], x_near) > self.cost(V[range(n + 1), :],
                                                                                                   P[range(n + 1)],
                                                                                                   x_new) + self.distance(x_new, x_near)):
                            idx = np.where((V[range(n), :] == x_near).all(axis=1))[0][0]
                            P[idx] = n
                if (x_new == self.x_goal).all():
                    success = True
                    reverse_path = []
                    current_state_index = n
                    while P[current_state_index] != -1:
                        reverse_path.append(V[current_state_index, :])
                        current_state_index = P[current_state_index]
                    reverse_path.append(self.x_init)
                    self.path = list(reversed(reverse_path))
                    break
                n = n + 1

        if success:
            print("Solution found!")
        else:
            print("Solution not found!")
        return success
