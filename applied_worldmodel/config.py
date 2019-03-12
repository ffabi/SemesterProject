import numpy as np
import random
# Actions: Steering: real valued in [-1, 1] Gas: real valued in [0, 1] Brake: real valued in [0, 1]
def generate_data_action(t, current_action):
    
    # Speeding up the car after the start
    if t < 60:
        return np.array([0,1,0])
    
    # Choose a random action every 5 frame
    if t % 5 > 0:
        return current_action
    
    rn = random.randint(0,9)
    if rn in [0]:
        # Do nothing
        return np.array([0,0,0])
    if rn in [1,2,3,4]:
        # Random gas
        return np.array([0,random.random(),0])
    if rn in [5,6,7]:
        # Random steer left
        return np.array([-random.random(),0,0]) # no gas at steering -> no drifting
    if rn in [8]:
        # Random steer right
        return np.array([random.random(),0,0])
    if rn in [9]:
        # Random brake
        return np.array([0,0,random.random()])


def adjust_obs(obs):
    return obs.astype('float32') / 255.

train_envs = ['car_racing']
test_envs = ['car_racing']