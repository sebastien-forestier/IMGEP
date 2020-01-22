from __future__ import print_function

import numpy as np

from explauto.interest_model.random import RandomInterest

from dataset import Dataset


# Adapted from explauto.interest_model.random, with a max number of points in the dataset
class MiscRandomInterest(RandomInterest):
    def __init__(self, conf, expl_dims, n_sdims, win_size=100):
        
        RandomInterest.__init__(self, conf, expl_dims)
        
        self.win_size = win_size
        self.n_sdims = n_sdims
        
        self.current_progress = 0.
        self.current_interest = 0.
        self.data = Dataset(len(expl_dims),
                            len(expl_dims),
                            max_size=1000)
    
    def competence_dist(self, target, reached):
        #d = np.linalg.norm(target[:len(reached)] - reached) / np.sqrt(len(reached)/len(self.expl_dims))
        if len(reached) < len(target):
            augm = np.tile(reached[-self.n_sdims:], (len(target) - len(reached)) // self.n_sdims)
            reached_augmented = np.hstack((reached, augm))
            d = np.linalg.norm(target - reached_augmented)
        else:
            d = np.linalg.norm(target - reached)
        return - d / self.n_sdims
        
    def update(self, sg, s, log=False):
        if len(self.data) > 0:
            # Current competence: from distance between current goal and reached s 
            c = self.competence_dist(sg, s)
            # Get NN of goal sg
            idx_sg_NN = self.data.nn_y(sg)[1][0]
            # Get corresponding reached sensory state s
            s_NN = self.data.get_x(idx_sg_NN)
            # old competence! from distance between current goal and old reached s for NN goal sg_NN 
            c_old = self.competence_dist(sg, s_NN)
            # Progress is the difference between current and old competence
            progress = c - c_old
            
            if log:
                print("\nsg  ", list(np.array(100*np.array(sg), dtype=np.int)))
                print("s   ", list(np.array(100*np.array(s), dtype=np.int)))
                print("s_NN", list(np.array(100*np.array(s_NN), dtype=np.int)))
                print("c", 0.01 * int(c * 100.))
                print("c_old", 0.01 * int(c_old * 100.))
                print("progress", 0.01 * int(progress * 100.))
                #print("data", self.data.data)
        else:
            progress = 0.
        # Update progress and interest
        self.current_progress += (1. / self.win_size) * (progress - self.current_progress)
        self.current_interest = abs(self.current_progress)
        # Log reached point and goal
        self.data.add_xy(s, sg)
        #print("Adding s=", s)
        return progress
    