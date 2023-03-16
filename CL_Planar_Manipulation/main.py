import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import delta_array_utils.RealDeltaControl as RDC

class MotionPlanner():
    def __init__(self, active_robots, planner_path):
        self.active_robots = active_robots
        self.delta_env = RDC.DeltaRobotEnv(active_robots=active_robots)
        self.delta_env.setup_delta_agents()
        df = pd.read_csv(planner_path, header=None)
        # df.replace(' nan', -7777.0, inplace=True)
        df = df.to_numpy(dtype=np.float32)
        
        # Motion Planner paths exclude last 7 values which describe
        # the 7D pose of object
        self.mp_df = df[:, :-7]
        self.num_modes = len(self.mp_df)/5
        print("Num Modes: ", self.num_modes)
        self.obj_pose_df = df[:, -7:]


        # self.nan_array = np.array((-7777.0,-7777.0,-7777.0))
        # self.nan_array_substitute = np.array((0.0,0.0,5.5))

    def execute_plan(self, init_finga):
        modes = np.array_split(self.mp_df, self.num_modes)
        obj_poses = np.array_split(self.obj_pose_df, self.num_modes)
        
        if init_finga is not None:
            self.delta_env.set_init_finga(init_finga)
            x = input("Press Enter to continue...")

        self.delta_env.set_init_plan(obj_pose = obj_poses[0][0])
        
        for i in range(int(self.num_modes)):
            print("MATATA Mode: ", i)
            traj = modes[i]
            if i != int(self.num_modes)-1:
                t2 = modes[i+1]
                self.delta_env.set_plan_horizontal(traj.reshape((5,len(self.active_robots),6)), obj_poses[i], traj2 = t2.reshape((5,len(self.active_robots),6))[0])
            else:
                self.delta_env.set_plan_horizontal(traj.reshape((5,len(self.active_robots),6)), obj_poses[i], traj2 = None)

    
    def execute_plan_inhand(self):
        modes = np.array_split(self.mp_df, self.num_modes)
        obj_poses = np.array_split(self.obj_pose_df, self.num_modes)

        self.delta_env.set_init_plan(obj_pose = obj_poses[0][0])
        
        for i in range(int(self.num_modes)):
            print("MATATA Mode: ", i)
            traj = modes[i]
            if i != int(self.num_modes)-1:
                t2 = modes[i+1]
                self.delta_env.set_plan_inhand(traj.reshape((5,len(self.active_robots),6)), obj_poses[i], traj2 = t2.reshape((5,len(self.active_robots),6))[0])
            else:
                self.delta_env.set_plan_inhand(traj.reshape((5,len(self.active_robots),6)), obj_poses[i], traj2 = None)

    def execute_plan_vertical(self, init_finga):
        modes = np.array_split(self.mp_df, self.num_modes)
        obj_poses = np.array_split(self.obj_pose_df, self.num_modes)
        for i in range(int(self.num_modes))[:3]:
            traj = modes[i]
            for traj in traj.reshape((5,len(self.active_robots),6)):
                for pos in traj[2:]:
                    print(f"Finger Poses: {pos[:2]}")
            # if i != int(self.num_modes)-1:
            #     t2 = modes[i+1]
            #     self.delta_env.set_plan(traj.reshape((5,len(self.active_robots),6)), obj_poses[i], traj2 = t2.reshape((5,len(self.active_robots),6))[0])
            # else:
            #     self.delta_env.set_plan(traj.reshape((5,len(self.active_robots),6)), obj_poses[i], traj2 = None)


    def test_traj(self):
        robot_poses = []
        for i in self.active_robots:
            robot_poses.append(self.delta_env.RC.robot_positions[i])
        # movement = np.array([20, 0])
        movement = np.array([0, 0])
        final_poses = np.array(robot_poses) + movement
        final_poses = final_poses.reshape((1,len(self.active_robots),2))
        self.delta_env.set_plan(final_poses/10)

        movement = np.array([10, 0])
        final_poses[0,:1] = np.array(robot_poses[:1]) + movement
        
        final_poses[0,1:] = np.array(robot_poses[1:]) - movement

        # final_poses = final_poses.reshape((1,len(self.active_robots),2))
        self.delta_env.set_plan(final_poses/10)
            

            

if __name__=="__main__":
    env = RDC.DeltaRobotEnv(active_robots=[-1])
    env.setup_delta_agents(np.array([0,0]))
    print("Done")
    """ 
    Make sure active robots list is in correct order of the rows of the motion planner
    """
    """ VIDEO DONE"""
    # motion_planner = MotionPlanner([(0,0),(0,2),(0,4),(2,0),(2,2),(2,4)], "./MCTS/data/delta_array/plan_results/actually_good_6finger_pushing_2.csv")
    # motion_planner.execute_plan(init_finga = [(0,0),(2,0)])

    """ VIDEO DONE """
    # motion_planner = MotionPlanner([(0,0),(0,2),(2,0),(2,2)], "./MCTS/data/delta_array/plan_results/actually_good_6finger_pushing_2.csv")
    # motion_planner.execute_plan(init_finga = [(0,0),(2,0)])
    
    """ VIDEO DONE: 5 finger rot """
    # motion_planner = MotionPlanner([(1,1),(1,2),(1,3),(2,1),(2,2)], "./MCTS/data/delta_array/plan_results/planar_manipulation_ouput_rotation_block.csv")
    # motion_planner.execute_plan(init_finga = None)
    
    """ 5 finger in hand """
    # motion_planner = MotionPlanner([(1,1),(1,2),(1,3),(2,1),(2,2)], "./MCTS/data/delta_array/plan_results/planar_manipulation_ouput.csv")
    # motion_planner.execute_plan_inhand()

    # motion_planner = MotionPlanner([(0,1),(0,2),(1,1),(1,3),(2,1),(2,2)], "./MCTS/data/delta_array/plan_results/good_big_block_rotation.csv")
    """ VIDEO DONE  """
    # motion_planner = MotionPlanner([(0,1),(0,2),(1,1),(1,3),(2,1),(2,2)], "./MCTS/data/delta_array/plan_results/planar_manipulation_ouput.csv")
    # motion_planner.execute_plan(init_finga = None)



    # motion_planner.test_traj()