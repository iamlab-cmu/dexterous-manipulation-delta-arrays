To set up the environment in conda:

- Create a new conda env with: conda create -n <name> python=3.8
- Activate the env with: conda activate <name>
- Go to isaacgym preview 4 https://developer.nvidia.com/isaac-gym/download, and download the file.
- Extract the file, and in isaacgym/python folder run "pip install -e ."
- Clone isaacgym utils https://github.com/iamlab-cmu/isaacgym-utils, cd into it and install with the same command as above.
- cd back into this repo and run "pip install -r requirements.txt"

Understanding the structure:

config/env.yaml has the env parameters that are read into main.py. 
main.py sets up the isaacgym environment, RL agent(s), and the DeltaArraySim object called "fingers".

The "run" function in main.py determines the function to use for policy in delta_array_sim.py.

