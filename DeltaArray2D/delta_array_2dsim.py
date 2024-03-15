from typing import Dict

import torch
from torch import Tensor

from vmas import render_interactively
from vmas.simulator.core import Agent, World, Landmark, Sphere, Box
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, X, TorchUtils
from vmas.simulator.controllers.velocity_controller import VelocityController

class DeltaArray2DSim(BaseScenario):
    def make_world(self, batch_dim:int, device:torch.device, **kwargs):
        world = World(batch_dim, device, substeps=1, contact_margin=1e-4) # Increase substeps if simulation seems unstable.
        self.shaping_factor = 100
        
        self.robot_positions = torch.zeros((64,2))
        for i in range(8):
            for j in range(8):
                if i%2!=0:
                    finger_pos = torch.tensor((i*0.0375, j*0.043301 - 0.02165))
                else:
                    finger_pos = torch.tensor((i*0.0375, j*0.043301))
                self.robot_positions[i*8 + j] = finger_pos
                agent = Agent(name=f"agent_{i}_{j}", shape=Sphere(0.0075), u_range=0.03, collide=False, color=Color.RED)
                world.add_agent(agent)
        

        goal = Landmark(name="goal", collide=False, shape=Sphere(0.0075), color=Color.GRAY)
        world.add_landmark(goal)
        self.disc = Landmark(name="disc", collide=True, shape=Sphere(0.035), color=Color.GREEN)
        self.disc.goal = goal
        world.add_landmark(self.disc)
        return world

    def reset_world_at(self, env_index=None):
        dim = 1 if env_index is not None else self.world.batch_dim
        a = torch.zeros((dim), device=self.world.device, dtype=torch.float32).uniform_(0.02, 0.24)
        b = torch.zeros((dim), device=self.world.device, dtype=torch.float32).uniform_(0.008, 0.27)
        pos = torch.stack((a,b), axis=-1)
        self.disc.set_pos(pos, batch_index=env_index)

        for n, agent in enumerate(self.world.agents):
            agent.set_pos(self.robot_positions[n].repeat(dim,1), batch_index=env_index)

        self.disc.goal.set_pos(pos + torch.zeros((dim, 2), device=self.world.device, dtype=torch.float32).uniform_(-0.02, 0.02),batch_index=env_index)

        if env_index is None:
            self.disc.global_shaping = (torch.linalg.vector_norm(self.disc.state.pos - self.disc.goal.state.pos, dim=1) * self.shaping_factor)
            self.disc.on_goal = torch.zeros(self.world.batch_dim, dtype=torch.bool, device=self.world.device)
        else:
            self.disc.global_shaping[env_index] = (
                torch.linalg.vector_norm(self.disc.state.pos[env_index] - self.disc.goal.state.pos[env_index]) * self.shaping_factor
            )
            self.disc.on_goal[env_index] = False

    def reward(self, agent):
        if agent == self.world.agents[0]:
            self.rew = torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.float32)
            self.disc.dist_to_goal = torch.linalg.vector_norm(self.disc.state.pos - self.disc.goal.state.pos, dim=1)
        return self.rew

    def observation(self, agent:Agent):
        return torch.cat((agent.state.pos, self.disc.state.pos, torch.linalg.vector_norm(self.disc.goal.state.pos - self.disc.state.pos, dim=1, keepdim=True)), dim=-1)

    def done(self):
        return self.disc.on_goal