from environment import ArmToolsToysEnvironment
import numpy as np
import time


gui = True # Plot environment animation ?
n = 10000 # Number of iterations

env = ArmToolsToysEnvironment()

t = time.time()

stick1_moved = 0
stick2_moved = 0
obj1_moved = 0
obj2_moved = 0


for i in range(n):
    n_dims = 4
    n_basis = 5
    m = np.random.random(n_dims * n_basis) * 2. - 1.

    s = env.update(m, gui=gui)
    
    stick1_moved += env.stick1_held
    stick2_moved += env.stick2_held
    obj1_moved += env.magnet1_move
    obj2_moved += env.scratch1_move
    
print("n stick1_moved", stick1_moved)
print("n stick2_moved", stick2_moved)
print("n obj1_moved", obj1_moved)
print("n obj2_moved", obj2_moved)

print("\nMovements:", n, "Steps:", 5 * n * env.steps_per_basis, "Time:", time.time() - t, "Time per step:", (time.time() - t) / (5 * n * env.steps_per_basis))
    