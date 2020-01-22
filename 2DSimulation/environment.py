import numpy as np
import matplotlib.pyplot as plt

from explauto.environment.environment import Environment


class GRBFTrajectory(object):
    def __init__(self, n_dims, sigma, steps_per_basis, max_basis):
        self.n_dims = n_dims
        self.sigma = sigma
        self.alpha = - 1. / (2. * self.sigma ** 2.)
        self.steps_per_basis = steps_per_basis
        self.max_basis = max_basis
        self.precomputed_gaussian = np.zeros(2 * self.max_basis * self.steps_per_basis)
        for i in range(2 * self.max_basis * self.steps_per_basis):
            self.precomputed_gaussian[i] = self.gaussian(self.max_basis * self.steps_per_basis, i)
        
    def gaussian(self, center, t):
        return np.exp(self.alpha * (center - t) ** 2.)
    
    def trajectory(self, weights):
        n_basis = len(weights)//self.n_dims
        weights = np.reshape(weights, (n_basis, self.n_dims)).T
        steps = self.steps_per_basis * n_basis
        traj = np.zeros((steps, self.n_dims))
        for step in range(steps):
            g = self.precomputed_gaussian[self.max_basis * self.steps_per_basis + self.steps_per_basis - 1 - step::self.steps_per_basis][:n_basis]
            traj[step] = np.dot(weights, g)
        return np.clip(traj, -1., 1.)
    
    def plot(self, traj):
        plt.plot(traj)
        plt.ylim([-1.05, 1.05])


class ArmToolsToysEnvironment(Environment):
    use_process = True

    def __init__(self, rdm_distractors=True):
        
        self.rdm_distractors = rdm_distractors
        
        self.viewer = None
        self.lines = None
        
        
        self.motor_dims = 4
        self.sensory_dims = 31
        self.steps_per_basis = 10
        self.sigma = self.steps_per_basis // 2
        self.max_basis = 5
        
        self.trajectory_generator = GRBFTrajectory(self.motor_dims, 
                                                   self.sigma,
                                                   self.steps_per_basis,
                                                   self.max_basis)
        
            
        self.m_mins=[-1.] * self.motor_dims * self.max_basis
        self.m_maxs=[1.] * self.motor_dims * self.max_basis
        self.s_mins=([-1] * 3 + [-1.5] * 28) * self.max_basis
        self.s_maxs=([1] * 3 + [1.5] * 28) * self.max_basis
        
        Environment.__init__(self, 
                             self.m_mins, 
                             self.m_maxs, 
                             self.s_mins, 
                             self.s_maxs)


        self.n_act = self.motor_dims
        self.n_obs = 31
        self.epsilon = 0.1
        self.n_timesteps = 50
        self.n_step = 0
        
        # GripArm
        self.arm_lengths = [0.5, 0.3, 0.2]
        self.arm_angle_shift = 0.5
        self.arm_rest_state = [0., 0., 0., 0.]

        # Stick1
        self.stick1_length = 0.5
        self.stick1_type = "magnetic"
        self.stick1_handle_tol = 0.03
        self.stick1_handle_tol_sq = self.stick1_handle_tol ** 2.
        self.stick1_rest_state = [-0.75, 0.25, 0.75]
        
        # Stick2
        self.stick2_length = 0.5
        self.stick2_type = "scratch"
        self.stick2_handle_tol = 0.03
        self.stick2_handle_tol_sq = self.stick1_handle_tol ** 2.
        self.stick2_rest_state = [0.75, 0.25, 0.25]
        
        # Magnet1
        self.magnet1_tolsq = 0.01 ** 2.
        self.magnet1_rest_state = [-0.3, 1.1]
        # Magnet2
        self.magnet2_tolsq = 0.
        self.magnet2_rest_state = [-0.5, 1.5]
        # Magnet3
        self.magnet3_tolsq = 0.
        self.magnet3_rest_state = [-0.3, 1.5]
        
        # Scratch1
        self.scratch1_tolsq = 0.01 ** 2.
        self.scratch1_rest_state = [0.3, 1.1]
        # Scratch2
        self.scratch2_tolsq = 0.
        self.scratch2_rest_state = [0.3, 1.5]
        # Scratch3
        self.scratch3_tolsq = 0.
        self.scratch3_rest_state = [0.5, 1.5]
        
        # Cat
        self.cat_noise = 0.1
        self.cat_rest_state = [-0.1, 1.1]
        # Dog
        self.dog_noise = 0.1
        self.dog_rest_state = [0.1, 1.1]

        # Static objects
        self.static_objects_rest_state = [[-0.7, 1.1],
                                          [-0.5, 1.1],
                                          [0.5, 1.1],
                                          [0.7, 1.1]]
        
        
        self.n_stick1_moved = 0
        self.n_stick2_moved = 0
        self.n_obj1_moved = 0
        self.n_obj2_moved = 0
        
        self.reset()
    
    def reset(self):
        
        if self.n_step:
            self.n_stick1_moved += self.stick1_moved
            self.n_stick2_moved += self.stick2_moved
            self.n_obj1_moved += self.magnet1_move
            self.n_obj2_moved += self.scratch1_move
        
        self.n_step = 0
        self.arm_angles = self.arm_rest_state[:-1]
        a = self.arm_angle_shift + np.cumsum(np.array(self.arm_angles))
        a_pi = np.pi * a
        self.hand_pos = np.array([np.sum(np.cos(a_pi)*self.arm_lengths), np.sum(np.sin(a_pi)*self.arm_lengths)])
        
        self.gripper = self.arm_rest_state[3]
        
        self.stick1_held = False
        self.stick1_moved = False
        self.stick1_handle_pos = np.array(self.stick1_rest_state[0:2])
        self.stick1_angle = self.stick1_rest_state[2]
        
        self.stick2_held = False
        self.stick2_moved = False
        self.stick2_handle_pos = np.array(self.stick2_rest_state[0:2])
        self.stick2_angle = self.stick2_rest_state[2]
        
        a = np.pi * self.stick1_angle
        self.stick1_end_pos = [self.stick1_handle_pos[0] + np.cos(a) * self.stick1_length, 
                               self.stick1_handle_pos[1] + np.sin(a) * self.stick1_length]

        a = np.pi * self.stick2_angle
        self.stick2_end_pos = [self.stick2_handle_pos[0] + np.cos(a) * self.stick2_length, 
                               self.stick2_handle_pos[1] + np.sin(a) * self.stick2_length]
        
        self.magnet1_move = 0
        self.magnet1_pos = self.magnet1_rest_state
        self.magnet2_move = 0
        self.magnet2_pos = self.magnet2_rest_state
        self.magnet3_move = 0
        self.magnet3_pos = self.magnet3_rest_state
        
        self.scratch1_move = 0
        self.scratch1_pos = self.scratch1_rest_state
        self.scratch2_move = 0
        self.scratch2_pos = self.scratch2_rest_state
        self.scratch3_move = 0
        self.scratch3_pos = self.scratch3_rest_state
        
        self.static_objects_pos = list(self.static_objects_rest_state)
        
        self.cat_pos = np.array(self.cat_rest_state)
        self.dog_pos = np.array(self.dog_rest_state)
        
    def observe(self):
        return [
                self.hand_pos[0],
                self.hand_pos[1],
                self.gripper,
                self.stick1_end_pos[0],
                self.stick1_end_pos[1],
                self.stick2_end_pos[0],
                self.stick2_end_pos[1],
                self.magnet1_pos[0],
                self.magnet1_pos[1],
                self.magnet2_pos[0],
                self.magnet2_pos[1],
                self.magnet3_pos[0],
                self.magnet3_pos[1],
                self.scratch1_pos[0],
                self.scratch1_pos[1],
                self.scratch2_pos[0],
                self.scratch2_pos[1],
                self.scratch3_pos[0],
                self.scratch3_pos[1],
                self.cat_pos[0],
                self.cat_pos[1],
                self.dog_pos[0],
                self.dog_pos[1],
                self.static_objects_rest_state[0][0],
                self.static_objects_rest_state[0][1],
                self.static_objects_rest_state[1][0],
                self.static_objects_rest_state[1][1],
                self.static_objects_rest_state[2][0],
                self.static_objects_rest_state[2][1],
                self.static_objects_rest_state[3][0],
                self.static_objects_rest_state[3][1]
                ]
        
    def step(self, m):
        """Run one timestep of the environment's dynamics.
        """

        # GripArm
        self.arm_angles = m[:-1]
        # We optimize runtime
        #a = self.arm_angle_shift + np.cumsum(self.arm_angles)
        #a_pi = np.pi * a
        a = [self.arm_angle_shift + m[0]] * 3
        a[1] += m[1]
        a[2] = a[1] + m[2]
        #a_pi = np.pi * np.array(a)
        a_pi = [np.pi * a[0], 
                np.pi * a[1], 
                np.pi * a[2]]
        #self.hand_pos = [np.sum(np.cos(a_pi)*self.arm_lengths), np.sum(np.sin(a_pi)*self.arm_lengths)]
        self.hand_pos[0] = np.cos(a_pi[0])*self.arm_lengths[0]
        self.hand_pos[0] += np.cos(a_pi[1])*self.arm_lengths[1]
        self.hand_pos[0] += np.cos(a_pi[2])*self.arm_lengths[2]
        self.hand_pos[1] = np.sin(a_pi[0])*self.arm_lengths[0]
        self.hand_pos[1] += np.sin(a_pi[1])*self.arm_lengths[1]
        self.hand_pos[1] += np.sin(a_pi[2])*self.arm_lengths[2]
        if m[-1] >= 0.:
            new_gripper = 1. 
        else:
            new_gripper = -1.
        gripper_change = (self.gripper - new_gripper) / 2.
        self.gripper = new_gripper
        hand_angle = np.mod(a[-1] + 1, 2) - 1
        
        # Stick1
        if not self.stick1_held:
            if gripper_change == 1. and (self.hand_pos[0] - self.stick1_handle_pos[0]) ** 2. + (self.hand_pos[1] - self.stick1_handle_pos[1]) ** 2. < self.stick1_handle_tol_sq:
                self.stick1_handle_pos = list(self.hand_pos)
                self.stick1_angle = hand_angle#np.mod(hand_angle + self.stick1_handle_noise * np.random.randn() + 1, 2) - 1
                a = np.pi * self.stick1_angle
                self.stick1_end_pos = [self.stick1_handle_pos[0] + np.cos(a) * self.stick1_length, 
                                       self.stick1_handle_pos[1] + np.sin(a) * self.stick1_length]
                self.stick1_held = True
                self.stick1_moved = True
        else:
            if gripper_change == 0:
                self.stick1_handle_pos = list(self.hand_pos)
                self.stick1_angle = hand_angle#np.mod(hand_angle + self.stick1_handle_noise * np.random.randn() + 1, 2) - 1
                a = np.pi * self.stick1_angle
                self.stick1_end_pos = [self.stick1_handle_pos[0] + np.cos(a) * self.stick1_length, 
                                       self.stick1_handle_pos[1] + np.sin(a) * self.stick1_length]
            else:
                self.stick1_held = False
                
        # Stick2
        if not self.stick2_held:
            if gripper_change == 1. and (self.hand_pos[0] - self.stick2_handle_pos[0]) ** 2. + (self.hand_pos[1] - self.stick2_handle_pos[1]) ** 2. < self.stick2_handle_tol_sq:
                self.stick2_handle_pos = list(self.hand_pos)
                self.stick2_angle = hand_angle#np.mod(hand_angle + self.stick2_handle_noise * np.random.randn() + 1, 2) - 1
                a = np.pi * self.stick2_angle
                self.stick2_end_pos = [self.stick2_handle_pos[0] + np.cos(a) * self.stick2_length, 
                                       self.stick2_handle_pos[1] + np.sin(a) * self.stick2_length]
                self.stick2_held = True
                self.stick2_moved = True
        else:
            if gripper_change == 0:
                self.stick2_handle_pos = list(self.hand_pos)
                self.stick2_angle = hand_angle#np.mod(hand_angle + self.stick2_handle_noise * np.random.randn() + 1, 2) - 1
                a = np.pi * self.stick2_angle
                self.stick2_end_pos = [self.stick2_handle_pos[0] + np.cos(a) * self.stick2_length, 
                                       self.stick2_handle_pos[1] + np.sin(a) * self.stick2_length]
            else:
                self.stick2_held = False

        # Magnet1
        if self.magnet1_move == 1 or (self.stick1_end_pos[0] - self.magnet1_pos[0]) ** 2 + (self.stick1_end_pos[1] - self.magnet1_pos[1]) ** 2 < self.magnet1_tolsq:
            self.magnet1_pos = self.stick1_end_pos[0:2]
            self.magnet1_move = 1
        # Scratch1
        if self.scratch1_move == 1 or (self.stick2_end_pos[0] - self.scratch1_pos[0]) ** 2 + (self.stick2_end_pos[1] - self.scratch1_pos[1]) ** 2 < self.scratch1_tolsq:
            self.scratch1_pos = self.stick2_end_pos[0:2]
            self.scratch1_move = 1

        if self.rdm_distractors:
            # Cat
            rdm = np.random.randn(4)
            self.cat_pos = self.cat_pos + rdm[:2] * self.cat_noise
            # Dog
            self.dog_pos = self.dog_pos + rdm[2:] * self.dog_noise
        
        self.n_step += 1
        
    def render(self, close=False):
        
        if self.viewer is None:
            self.start_viewer()
        
        fig = plt.gcf()
        ax = plt.gca()
        
        fig.canvas.restore_region(self.background)
        
        # Arm
        angles = np.array(self.arm_angles)
        angles[0] += self.arm_angle_shift
        a = np.cumsum(np.pi * angles)
        x = np.hstack((0., np.cumsum(np.cos(a)*self.arm_lengths)))
        y = np.hstack((0., np.cumsum(np.sin(a)*self.arm_lengths)))
        self.lines["l1"][0].set_data(x, y)
        self.lines["l2"][0].set_data(x[0], y[0])
        for i in range(len(self.arm_lengths)-1):
            self.lines[i][0].set_data(x[i+1], y[i+1])
        self.lines["l3"][0].set_data(x[-1], y[-1])

        
        # Gripper
        if self.gripper >= 0.:
            self.lines["g1"][0].set_data(x[-1], y[-1])
            self.lines["g2"][0].set_data(3, 3)
        else:
            self.lines["g1"][0].set_data(3, 3)
            self.lines["g2"][0].set_data(x[-1], y[-1])
            
        # Stick1
        if self.stick1_held or self.n_step <= 1:
            self.lines["s11"][0].set_data([self.stick1_handle_pos[0], self.stick1_end_pos[0]],
                                          [self.stick1_handle_pos[1], self.stick1_end_pos[1]])
            self.lines["s12"][0].set_data(self.stick1_handle_pos[0], 
                                          self.stick1_handle_pos[1])
            self.lines["s13"][0].set_data(self.stick1_end_pos[0],
                                          self.stick1_end_pos[1])
        
        # Magnet1
        self.patches['mag1'].set_xy((self.magnet1_pos[0] - 0.05, 
                                     self.magnet1_pos[1] - 0.05))
            
        # Stick2
        if self.stick2_held or self.n_step <= 1:
            self.lines["s21"][0].set_data([self.stick2_handle_pos[0], self.stick2_end_pos[0]],
                                          [self.stick2_handle_pos[1], self.stick2_end_pos[1]])
            self.lines["s22"][0].set_data(self.stick2_handle_pos[0], 
                                          self.stick2_handle_pos[1])
            self.lines["s23"][0].set_data(self.stick2_end_pos[0],
                                          self.stick2_end_pos[1])
            
        # Scratch1
        self.patches['scr1'].set_xy((self.scratch1_pos[0] - 0.05, 
                                     self.scratch1_pos[1] - 0.05))
    
        

        # Cat
        self.patches['cat'].set_xy((self.cat_pos[0] - 0.05, 
                                    self.cat_pos[1] - 0.05))
        # Dog
        self.patches['dog'].set_xy((self.dog_pos[0] - 0.05, 
                                    self.dog_pos[1] - 0.05))
            
        fig.canvas.blit(ax.bbox)
        #plt.pause(0.01)

    def start_viewer(self):
        self.viewer = plt.figure(figsize=(5, 5), frameon=False)
        fig = plt.gcf()
        ax = plt.gca()
        plt.axis('off')
        fig.show()
        fig.canvas.draw()
        self.lines = {}
        self.patches = {}
        
        # Arm
        angles = np.array(self.arm_angles)
        angles[0] += self.arm_angle_shift
        a = np.cumsum(np.pi * angles)
        x = np.hstack((0., np.cumsum(np.cos(a)*self.arm_lengths)))
        y = np.hstack((0., np.cumsum(np.sin(a)*self.arm_lengths)))
        self.lines["l1"] = ax.plot(x, y, 'grey', lw=4)
        self.lines["l2"] = ax.plot(x[0], y[0], 'ok', ms=8)
        for i in range(len(self.arm_lengths)-1):
            self.lines[i] = ax.plot(x[i+1], y[i+1], 'ok', ms=8)
        self.lines["l3"] = ax.plot(x[-1], y[-1], 'or', ms=4)
        ax.axis([-2, 2., -1.5, 2.])        

        # Gripper
        self.lines["g1"] = ax.plot(3, 3, 'o', markerfacecolor='none', markeredgewidth=3, markeredgecolor="r", ms=20)
        self.lines["g2"] = ax.plot(3, 3, 'o', color="r", ms=10)
            
        # Stick1
        self.lines["s11"] = ax.plot([self.stick1_handle_pos[0], self.stick1_end_pos[0]], [self.stick1_handle_pos[1], self.stick1_end_pos[1]], '-', color='grey', lw=4)
        self.lines["s12"] = ax.plot(self.stick1_handle_pos[0], self.stick1_handle_pos[1], 'o', color = "g", ms=6)
        self.lines["s13"] = ax.plot(self.stick1_end_pos[0], self.stick1_end_pos[1], 'o', color = "b", ms=6)
                    
        # Stick2
        self.lines["s21"] = ax.plot([self.stick2_handle_pos[0], self.stick2_end_pos[0]], [self.stick2_handle_pos[1], self.stick2_end_pos[1]], '-', color='grey', lw=4)
        self.lines["s22"] = ax.plot(self.stick2_handle_pos[0], self.stick2_handle_pos[1], 'o', color = "g", ms=6)
        self.lines["s23"] = ax.plot(self.stick2_end_pos[0], self.stick2_end_pos[1], 'o', color = "c", ms=6)
                
        # Magnet1
        p = plt.Rectangle((self.magnet1_pos[0] - 0.05, self.magnet1_pos[1] - 0.05), 0.1, 0.1, fc='b')
        self.patches['mag1'] = p
        ax.add_patch(p)
        # Magnet2
        ax.add_patch(plt.Rectangle((self.magnet2_pos[0] - 0.05, self.magnet2_pos[1] - 0.05), 0.1, 0.1, fc='b'))
        # Magnet3
        ax.add_patch(plt.Rectangle((self.magnet3_pos[0] - 0.05, self.magnet3_pos[1] - 0.05), 0.1, 0.1, fc='b'))
        
        # Scratch1
        p = plt.Rectangle((self.scratch1_pos[0] - 0.05, self.scratch1_pos[1] - 0.05), 0.1, 0.1, fc="c")
        ax.add_patch(p)
        self.patches['scr1'] = p
        # Scratch2
        ax.add_patch(plt.Rectangle((self.scratch2_pos[0] - 0.05, self.scratch2_pos[1] - 0.05), 0.1, 0.1, fc="c"))
        # Scratch3
        ax.add_patch(plt.Rectangle((self.scratch3_pos[0] - 0.05, self.scratch3_pos[1] - 0.05), 0.1, 0.1, fc="c"))

        # Cat
        p = plt.Rectangle((self.cat_pos[0] - 0.05, self.cat_pos[1] - 0.05), 0.1, 0.1, fc="m")
        ax.add_patch(p)
        self.patches['cat'] = p
        # Dog
        p = plt.Rectangle((self.dog_pos[0] - 0.05, self.dog_pos[1] - 0.05), 0.1, 0.1, fc="y")
        ax.add_patch(p)
        self.patches['dog'] = p
        
        # Static
        for pos in self.static_objects_pos:
            ax.add_patch(plt.Rectangle((pos[0] - 0.05, pos[1] - 0.05), 0.1, 0.1, fc='k'))
        
        self.background = fig.canvas.copy_from_bbox(ax.bbox)
        
    def close(self):
        if self.viewer is not None:
            plt.close(self.viewer)
            self.viewer = None
     
    def compute_motor_command(self, m_ag):
        return self.trajectory_generator.trajectory(m_ag)

    def compute_sensori_effect(self, m, gui=False):
        s = []
        for i in range(len(m)):
            self.step(m[i])
            if gui:
                self.render()
            if i % self.steps_per_basis == 0:
                s.append(list(self.observe()))
            
        s_ag = [item for si in s for item in si]
        #if len(s_ag) < self.conf.s_ndims:
        #    s_ag = s_ag + [0.] * (self.conf.s_ndims - len(s_ag))
        return s_ag
    
    def update(self, m_ag, reset=True, gui=False):
        if reset:
            self.reset()
        m = self.compute_motor_command(m_ag)
        s = self.compute_sensori_effect(m, gui)
        return s
    
    
    
