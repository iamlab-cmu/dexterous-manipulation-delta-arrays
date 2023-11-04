# A Goal-Conditioned Multi-Agent RL Project to Learn Cooperative Distributed Manipulation Policies Using the Delta Arrays.

## Stages of project evolution:
1. Simulated the delta array in IsaacGym to match the coordinate frames and the camera input of the real robot.
2. Implemented various policy gradient algorithms (A2C with PPO, DDPG, SAC) to learn to "touch" the boundary of the object using a single robot.
3. Found SAC to be the most robust for learning sample-efficient stochastic policies with the most stable updates. 
4. Learned policy in sim, works on the real soft robots - they're able to grasp the block.

## Next steps: 
1. Train a multi-agent policy to learn to move the block with *n* robots in it's neighborhood. 
2. Generate object-policies to generate attractors on the object surface to generate object trajectory that can be tracked by the learned policy.