import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import delta_array_utils.DeltaArrayControl as DAC

class MotionPlanner():
    def __init__(self, active_robots, planner_path):
        self.active_robots = active_robots
        self.delta_env = DAC.DeltaArrayEnv(active_robots=active_robots)
        self.delta_env.setup_delta_agents()
        
if __name__=="__main__":
    env = DAC.DeltaArrayEnv(active_robots=[ (7,0)])
    env.setup_delta_agents(np.array([0,0]))
    print("Done")
    
