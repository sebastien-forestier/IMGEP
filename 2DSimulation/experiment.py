from __future__ import print_function

import numpy as np
from environment import ArmToolsToysEnvironment
from explauto.utils import prop_choice
import time
from learning_module import LearningModule
import matplotlib.pyplot as plt
import cProfile
import pickle


class Experiment(object):
    def __init__(self, 
                 seed=0,
                 explo_noise=0.05, 
                 rmb_prop=0.1, 
                 optim_explo="full", 
                 n_explore=4, 
                 condition="RMB",
                 end_point=False,
                 distractors="both"):
        
        self.seed = seed
        self.explo_noise = explo_noise
        self.rmb_prop = rmb_prop
        self.optim_explo = optim_explo
        self.n_explore = n_explore
        self.condition = condition
        self.end_point = end_point
        self.distractors = distractors
        
        np.random.seed(self.seed)
        
            
        self.environment = ArmToolsToysEnvironment(
            rdm_distractors=distractors == "random" or distractors == "both")
        
        self.data = []
        self.interests_evolution = []
        self.explo = None
        self.chosen_modules = []
        self.steps = []
        self.i = 0
    
        # Define motor and sensory spaces:
        self.m_ndims = 4 # number of motor parameters
        self.s_ndims = 31 # number of sensory parameters
        self.max_steps = 5
        self.m_space = range(self.m_ndims)
        self.s_hand  = range(self.m_ndims, self.m_ndims+3)
        self.s_tool1  = range(self.m_ndims+3, self.m_ndims+5)
        self.s_tool2  = range(self.m_ndims+5, self.m_ndims+7)
        self.s_obj1  = range(self.m_ndims+7, self.m_ndims+9)
        self.s_obj2  = range(self.m_ndims+9, self.m_ndims+11)
        self.s_obj3  = range(self.m_ndims+11, self.m_ndims+13)
        self.s_obj4  = range(self.m_ndims+13, self.m_ndims+15)
        self.s_obj5  = range(self.m_ndims+15, self.m_ndims+17)
        self.s_obj6  = range(self.m_ndims+17, self.m_ndims+19)
        self.s_obj7  = range(self.m_ndims+19, self.m_ndims+21)
        self.s_obj8  = range(self.m_ndims+21, self.m_ndims+23)
        self.s_obj9  = range(self.m_ndims+23, self.m_ndims+25)
        self.s_obj10  = range(self.m_ndims+25, self.m_ndims+27)
        self.s_obj11  = range(self.m_ndims+27, self.m_ndims+29)
        self.s_obj12  = range(self.m_ndims+29, self.m_ndims+31)
    
        self.s_flat = range(self.m_ndims, self.m_ndims + 31)
        
        self.s_spaces = dict(s_hand  = range(self.m_ndims, self.m_ndims+3),
                            s_tool1  = range(self.m_ndims+3, self.m_ndims+5),
                            s_tool2  = range(self.m_ndims+5, self.m_ndims+7),
                            s_obj1  = range(self.m_ndims+7, self.m_ndims+9),
                            s_obj2  = range(self.m_ndims+9, self.m_ndims+11),
                            s_obj3  = range(self.m_ndims+11, self.m_ndims+13),
                            s_obj4  = range(self.m_ndims+13, self.m_ndims+15),
                            s_obj5  = range(self.m_ndims+15, self.m_ndims+17),
                            s_obj6  = range(self.m_ndims+17, self.m_ndims+19),
                            s_obj7  = range(self.m_ndims+19, self.m_ndims+21),
                            s_obj8  = range(self.m_ndims+21, self.m_ndims+23),
                            s_obj9  = range(self.m_ndims+23, self.m_ndims+25),
                            s_obj10  = range(self.m_ndims+25, self.m_ndims+27),
                            s_obj11  = range(self.m_ndims+27, self.m_ndims+29),
                            s_obj12  = range(self.m_ndims+29, self.m_ndims+31))
        
        # Create learning modules:
        self.learning_modules = {}
        if condition == "FRGB" or condition == "rmb":
            self.n_explore = 1
            self.n_test = 0
            self.learning_modules['mod1'] = LearningModule("mod1", self.m_space, self.s_flat, self.max_steps, 
                                                      self.environment.conf, explo_noise=explo_noise, optim_explo="full", end_point=end_point)
        elif condition == "SGS":
            self.n_explore = 1
            self.n_test = 0
            self.learning_modules['mod4'] = LearningModule("mod4", self.m_space, self.s_obj1, self.max_steps, self.environment.conf, explo_noise=explo_noise, optim_explo=optim_explo, end_point=end_point)
            
        else:
            self.n_test = 1
            
            self.learning_modules['mod1'] = LearningModule("mod1", self.m_space, self.s_hand, self.max_steps, self.environment.conf, explo_noise=explo_noise, optim_explo=optim_explo, end_point=end_point)
            self.learning_modules['mod2'] = LearningModule("mod2", self.m_space, self.s_tool1, self.max_steps, self.environment.conf, explo_noise=explo_noise, optim_explo=optim_explo, end_point=end_point)
            self.learning_modules['mod3'] = LearningModule("mod3", self.m_space, self.s_tool2, self.max_steps, self.environment.conf, explo_noise=explo_noise, optim_explo=optim_explo, end_point=end_point)
            
            self.learning_modules['mod4'] = LearningModule("mod4", self.m_space, self.s_obj1, self.max_steps, self.environment.conf, explo_noise=explo_noise, optim_explo=optim_explo, end_point=end_point)
            
            self.learning_modules['mod7'] = LearningModule("mod7", self.m_space, self.s_obj4, self.max_steps, self.environment.conf, explo_noise=explo_noise, optim_explo=optim_explo, end_point=end_point)
           
            if distractors == "random" or distractors == "both":
                self.learning_modules['mod10'] = LearningModule("mod10", self.m_space, self.s_obj7, self.max_steps, self.environment.conf, explo_noise=explo_noise, optim_explo=optim_explo, end_point=end_point)
                self.learning_modules['mod11'] = LearningModule("mod11", self.m_space, self.s_obj8, self.max_steps, self.environment.conf, explo_noise=explo_noise, optim_explo=optim_explo, end_point=end_point)
            
            if distractors == "static" or distractors == "both":
                self.learning_modules['mod5'] = LearningModule("mod5", self.m_space, self.s_obj2, self.max_steps, self.environment.conf, explo_noise=explo_noise, optim_explo=optim_explo, end_point=end_point)
                self.learning_modules['mod6'] = LearningModule("mod6", self.m_space, self.s_obj3, self.max_steps, self.environment.conf, explo_noise=explo_noise, optim_explo=optim_explo, end_point=end_point)
            
                self.learning_modules['mod8'] = LearningModule("mod8", self.m_space, self.s_obj5, self.max_steps, self.environment.conf, explo_noise=explo_noise, optim_explo=optim_explo, end_point=end_point)
                self.learning_modules['mod9'] = LearningModule("mod9", self.m_space, self.s_obj6, self.max_steps, self.environment.conf, explo_noise=explo_noise, optim_explo=optim_explo, end_point=end_point)
            
                self.learning_modules['mod12'] = LearningModule("mod12", self.m_space, self.s_obj9, self.max_steps, self.environment.conf, explo_noise=explo_noise, optim_explo=optim_explo, end_point=end_point)
                self.learning_modules['mod13'] = LearningModule("mod13", self.m_space, self.s_obj10, self.max_steps, self.environment.conf, explo_noise=explo_noise, optim_explo=optim_explo, end_point=end_point)
                self.learning_modules['mod14'] = LearningModule("mod14", self.m_space, self.s_obj11, self.max_steps, self.environment.conf, explo_noise=explo_noise, optim_explo=optim_explo, end_point=end_point)
                self.learning_modules['mod15'] = LearningModule("mod15", self.m_space, self.s_obj12, self.max_steps, self.environment.conf, explo_noise=explo_noise, optim_explo=optim_explo, end_point=end_point)
    
        self.avg_steps = 1.
    
        self.n_stick1_moved = 0
        self.n_stick2_moved = 0
        self.n_obj1_moved = 0
        self.n_obj2_moved = 0
        self.last_print = 0
    
    def execute_perceive(self, m):
        steps = len(m)//self.m_ndims
        s = self.environment.update(m) # execute this command and observe the corresponding sensory effect
        ms_array = np.zeros((steps, self.m_ndims + self.s_ndims), dtype=np.float16)
        ms_array[:,:self.m_ndims] = np.array(np.split(np.array(m), steps), dtype=np.float16)
        ms_array[:,self.m_ndims:] = np.array(np.split(np.array(s), steps), dtype=np.float16)
        #print "ms_array", ms_array, ms_array.shape
        # Update each sensorimotor models:
        for mid in self.learning_modules.keys():
            #print mid, self.learning_modules[mid].s_space, ms_array[:,self.learning_modules[mid].s_space]
            self.learning_modules[mid].update_sm(m, np.concatenate(ms_array[:,self.learning_modules[mid].s_space]))
                        
        self.data.append(ms_array)
        self.steps.append(steps)
        self.i += 1
        return ms_array, steps
    
            
    def compute_explo_space(self, s_space, n_checkpoints=1):
        data = np.array([self.data[i][-1,self.s_spaces[s_space]] for i in range(len(self.data))])
        mins = np.array([-1.5]*len(self.s_spaces[s_space]))
        maxs = np.array([1.5]*len(self.s_spaces[s_space]))
        checkpoints = [int(x) for x in np.linspace(len(data)/n_checkpoints, len(data), n_checkpoints)]
        n = len(mins)
        assert len(data[0]) == n
        gs = [0, 10, 100, 20, 10, 10, 10, 5, 5, 3][n]
        epss = (maxs - mins) / gs
        grid = np.zeros([gs] * n)
        #print np.size(grid), mins, maxs
        res = [0]
        for c in range(1, len(checkpoints)):
            for i in range(checkpoints[c-1], checkpoints[c]):
                idxs = np.array((data[i] - mins) / epss, dtype=int)
                #print c, i, idxs
                idxs[idxs>=gs] = gs-1
                idxs[idxs<0] = 0
                #print idxs
                grid[tuple(idxs)] = grid[tuple(idxs)] + 1
            grid[grid > 1] = 1
            res.append(np.sum(grid))
        return np.array(res) / gs ** n



        
    def run(self, iterations=100000, profile=False, print_logs=False):
        
        if profile:
            cp = cProfile.Profile()
            cp.enable()
    
        t = time.clock()
        while self.i < iterations:
            if print_logs:
                # Print number of iterations up to now:
                if self.i - self.last_print > 1000:
                    self.last_print = 1000 * (self.i // 1000)
                    print("\nIteration:", self.i)
                    print("Time:", int(10.*(time.clock() - t)) / 10.)
                    print("Average steps", int(10.*self.avg_steps) / 10.)
                    print("n_stick1_moved", self.environment.n_stick1_moved - self.n_stick1_moved)
                    print("n_stick2_moved", self.environment.n_stick2_moved - self.n_stick2_moved)
                    print("n_obj1_moved", self.environment.n_obj1_moved - self.n_obj1_moved)
                    print("n_obj2_moved", self.environment.n_obj2_moved - self.n_obj2_moved)
                    self.n_stick1_moved = self.environment.n_stick1_moved
                    self.n_stick2_moved = self.environment.n_stick2_moved
                    self.n_obj1_moved = self.environment.n_obj1_moved
                    self.n_obj2_moved = self.environment.n_obj2_moved
        
                    if self.condition == "AMB":
                        for mid in ["mod1", "mod2", "mod3", "mod4", "mod7", "mod10"]:
                            if mid in self.learning_modules:
                                print("Interest of module", mid, ":", int(1000.*self.learning_modules[mid].interest_model.current_interest) / 1000.)
        
                    t = time.clock()
    
            # Choose the babbling module (probabilities proportional to interests, with epsilon of random choice):
            if self.condition == "RMB":
                # Get the interest of modules
                interests = [self.learning_modules[mid].interest() for mid in self.learning_modules.keys()]
                self.interests_evolution.append(interests)
                babbling_module = np.random.choice(list(self.learning_modules.values()))
            elif self.condition == "AMB":
                # Get the interest of modules
                interests = [self.learning_modules[mid].interest() for mid in self.learning_modules.keys()]
                self.interests_evolution.append(interests)
                babbling_module = list(self.learning_modules.values())[prop_choice(interests, eps=0.2)]
                #babbling_module = self.learning_modules["mod1"]
            elif self.condition == "FRGB" or self.condition == "rmb":
                babbling_module = self.learning_modules["mod1"]            
            elif self.condition == "SGS":
                babbling_module = self.learning_modules["mod4"]            
            elif self.condition == "FC":
                fc = ["mod1", "mod2", "mod4", "mod3", "mod7"]
                m = self.i // (iterations // len(fc))
                babbling_module = self.learning_modules[fc[m]]
            else:
                raise NotImplementedError
    
            # The babbling module picks a random goal in its sensory space and returns 4 noisy motor commands:
            if babbling_module.t < babbling_module.motor_babbling_n_iter or np.random.random() < self.rmb_prop or self.condition == "rmb":
                m = babbling_module.motor_babbling(steps=1)
                ms_array, steps = self.execute_perceive(m)
                self.chosen_modules.append("random")
            else:
                self.chosen_modules.append(babbling_module.mid)
                sg = babbling_module.interest_model.sample()
                babbling_module.sg = sg
                for _ in range(self.n_explore):
                    m = babbling_module.inverse(sg)
                    ms_array, steps = self.execute_perceive(m)
                    self.avg_steps = self.avg_steps * 0.99 + 0.01 * steps
    
                # Update Interest
                if self.condition == "AMB":
                    m = babbling_module.inverse(sg, explore=False)
                    ms_array, steps = self.execute_perceive(m)
                    babbling_module.update_im(m, np.concatenate(ms_array[:,babbling_module.s_space]))
    
    
    
    
        if profile:
            cp.disable()
            cp.dump_stats("test.cprof")
    
        if print_logs:
            print("n stick1_moved", self.environment.n_stick1_moved)
            print("n stick2_moved", self.environment.n_stick2_moved)
            print("n obj1_moved", self.environment.n_obj1_moved)
            print("n obj2_moved", self.environment.n_obj2_moved)
        
            print()
            print("Parameters:", iterations, self.explo_noise, self.optim_explo, self.condition, self.distractors)
        


    def plot_interests(self):
    
        interests_evolution = np.array(self.interests_evolution)
        plt.plot(interests_evolution[:,0], label="hand")
        plt.plot(interests_evolution[:,1], label="stick1")
        plt.plot(interests_evolution[:,2], label="stick2")
        plt.plot(interests_evolution[:,3], label="toy1")
        plt.plot(interests_evolution[:,6], label="toy2")
        plt.plot(interests_evolution[:,-1], label="static")
        plt.plot(interests_evolution[:,9], label="cat")
    
        plt.legend()
    
    def compute_explo(self):
        self.explo = {}
        for s_space in ["s_hand", "s_tool1", "s_tool2", "s_obj1", "s_obj4"]:
            self.explo[s_space] = self.compute_explo_space(s_space, 10)
        
    def plot_explo(self):
        for s_space in ["s_hand", "s_tool1", "s_tool2", "s_obj1", "s_obj4"]:
            plt.plot(self.explo[s_space], label=s_space)
        plt.legend()
        
    def dump(self, filename):
        to_store = dict(
                      data=self.data,
                      steps=self.steps,
                      interests_evolution=self.interests_evolution,
                      explo=self.explo,
                      chosen_modules=self.chosen_modules
                      )
        for key in to_store.keys():
            with open(filename + "-" + key + ".pickle", 'wb') as f:
                pickle.dump(to_store[key], f)
#         
#     def load(self, filename):
#         with open(filename, 'r') as f:
#             logs = cPickle.load(f)
#             f.close()
#         self.data = logs["data"]
#         self.interests_evolution = logs["interests_evolution"]
#         
#         