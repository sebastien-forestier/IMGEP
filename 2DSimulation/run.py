import sys
import os
from experiment import Experiment

# Directory to write log file
log_dir = sys.argv[1]

# Pseudo-random seed
seed = int(sys.argv[2])

# Environment distractors ["none", "random", "static", "both"]
distractors = sys.argv[3]

# Optimize exploration ["random", "gaussian", "full"]
#full= FullMutation
#random= SSPM with random parameter after SS
#gaussian= SSPM with gaussian noise on parameters after SS.
optim_explo = sys.argv[4]

# Goal representation as end point: [True, False]
#True: goal = trajectory end point
#False: goal = trajectory samples
end_point = sys.argv[5] == "True"

# Experimental condition: ["rmb", "SGS", "FC", "FRGB", "RMB", "AMB"]
condition = sys.argv[6]

# Number of iterations
iterations = int(sys.argv[7])

# Exploration noise amplitude
explo_noise = 0.05

# Proportion of random motor babbling iterations
rmb_prop = 0.1

# Number of exploration iteration for each exploitation iteration
n_explore = 4 





xp = Experiment(seed=seed, 
                explo_noise=explo_noise,
                rmb_prop=rmb_prop,
                optim_explo=optim_explo, 
                n_explore=n_explore,
                condition=condition,
                end_point=end_point,
                distractors=distractors)

xp.run(iterations=iterations, print_logs=False)


filename = "-".join([str(arg) for arg in (seed,
                                          distractors,
                                          optim_explo,
                                          end_point,
                                          condition,
                                          iterations)])

xp.compute_explo()

if not os.path.exists(log_dir):
    os.mkdir(log_dir)
xp.dump(os.path.join(log_dir, filename))
