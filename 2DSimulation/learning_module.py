from __future__ import print_function

import numpy as np

from explauto.utils.config import make_configuration

from dataset import BufferedDataset
 
from interest_model import MiscRandomInterest


class LearningModule(object):
    def __init__(self, mid, m_space, s_space, max_steps, env_conf,
                 explo_noise=0.05, motor_babbling_n_iter=10,
                 optim_explo=None, end_point=False):

        
        self.mid = mid
        self.m_space = m_space
        self.s_space = s_space
        self.n_mdims = len(self.m_space)
        self.n_sdims = len(self.s_space)
        self.max_steps = max_steps
        self.env_conf = env_conf
        self.explo_noise = explo_noise
        self.motor_babbling_n_iter = motor_babbling_n_iter
        self.optim_explo = optim_explo
        self.end_point = end_point
        
        self.s = None
        self.sg = None
        self.last_interest = 0
        self.t = 0
        
        
        # Sensorimotor Model
        conf = make_configuration(list(env_conf.m_mins[m_space]) * self.max_steps,
                                   list(env_conf.m_maxs[m_space]) * self.max_steps,
                                   list(np.array(list(env_conf.m_mins[m_space]) + list(env_conf.s_mins))[s_space]) * self.max_steps,
                                   list(np.array(list(env_conf.m_maxs[m_space]) + list(env_conf.s_maxs))[s_space]) * self.max_steps)
            
        self.sm = BufferedDataset(conf.m_ndims, 
                                  conf.s_ndims,
                                  buffer_size=10000, # Size of a small kdtree buffer to update this one often and move the data to the big kdtree less often  
                                  lateness=100) # The model can be "late" by this number of points: they are not yet taken into account (added to the small kdtree)
        

        if self.end_point:
            self.sm_end = BufferedDataset(conf.m_ndims, 
                                          len(s_space),
                                          buffer_size=10000,
                                          lateness=100)
            
            self.interest_model = MiscRandomInterest(conf, 
                                                     conf.s_dims[-self.n_sdims:], 
                                                     self.n_sdims, 
                                                     win_size=200)
        else:
            self.interest_model = MiscRandomInterest(conf, 
                                                     conf.s_dims, 
                                                     self.n_sdims, 
                                                     win_size=200)
        
        
        
    def motor_babbling(self, steps=None):
        return np.random.random(self.n_mdims * self.max_steps) * 2. - 1.
        
    def inverse(self, sg, explore=True, log=False):
        # Get nearest neighbor
        if len(self.sm):
            if self.end_point:
                _, idx = self.sm_end.nn_y(sg[-len(self.s_space):])
                m = np.array(self.sm_end.get_x(idx[0]))
                snn = self.sm.get_y(idx[0])
            else:
                _, idx = self.sm.nn_y(sg)
                m = np.array(self.sm.get_x(idx[0]))
                snn = self.sm.get_y(idx[0])
        else:
            return self.motor_babbling()
        # Add Exploration Noise
        if explore:
            if self.optim_explo == "gaussian" or self.optim_explo == "random":
                # Detect Movement
                snn_steps = len(snn) // self.n_sdims
                move_step = snn_steps
                for i in range(1, snn_steps):
                    if abs(snn[self.n_sdims * i] - snn[self.n_sdims * (i-1)]) > 0.01:
                        #Move at step i
                        move_step = i
                        break
                # Explore after Movement detection
                if move_step == 1 or move_step == snn_steps:
                    start_explo = 0
                else:
                    start_explo = move_step
                
                if self.optim_explo == "gaussian":
                    explo_vect = [0.] * start_explo * self.n_mdims + [self.explo_noise]*(snn_steps-start_explo) * self.n_mdims
                    m = np.random.normal(m, explo_vect).clip(-1.,1.)
                else:
                    rdm = 2. * np.random.random(self.max_steps * self.n_mdims) - 1.
                    m[start_explo * self.n_mdims:] = rdm[start_explo * self.n_mdims:]
            
            elif self.optim_explo == "full":
                explo_vect = [self.explo_noise]*len(m)
                m = np.random.normal(m, explo_vect).clip(-1.,1.)
            else:
                raise NotImplementedError
        return m
            
    def produce(self):
        if self.t < self.motor_babbling_n_iter:
            self.m = self.motor_babbling()
            self.sg = None
        else:
            self.sg = self.interest_model.sample()
            self.m = self.inverse(self.sg)
        return self.m        
    
    def update_sm(self, m, s):
        if s[1] != s[-1]:
            self.sm.add_xy(m, s)
            if self.end_point:
                self.sm_end.add_xy(m, s[-len(self.s_space):])
        self.t += 1 
    
    def update_im(self, m, s):
        if self.t > self.motor_babbling_n_iter:
            if self.end_point:
                return self.interest_model.update(self.sg[-len(self.s_space):], s[-len(self.s_space):])
            else:
                return self.interest_model.update(self.sg, s)
        
    def interest(self): return self.interest_model.current_interest

    def perceive(self, m, s):
        self.update_sm(m, s)

    def get_m(self, ms): return ms[self.m_space]
    def get_s(self, ms): return ms[self.s_space]
    