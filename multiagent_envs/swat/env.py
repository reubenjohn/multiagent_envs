import cv2
import numpy as np

from multiagent_envs.env import MultiAgentEnv2d
from multiagent_envs.geometry import Point
from multiagent_envs.negotiator import DemandJudge


class AgentState(object):
	def __init__(self, pos: Point):
		self.pos = pos


class SWAT(MultiAgentEnv2d, DemandJudge):
	def __init__(self, n_agents: int, w=1280, h=720):
		super().__init__(n_agents, w, h)
		self.agent_states = [AgentState(point) for point in self.sample_unique_points(n_agents)]

	def sample_unique_points(self, size):
		return [self.pos_from_seed(seed) for seed in np.random.choice(range(self.w * self.h), size, replace=False)]

	def pos_from_seed(self, seed: int):
		return Point(seed % self.w, int(seed / self.w))

	def step(self, actions):
		super().step(actions)
		nones = [None for _ in range(self.n_agents)]
		if self.steps < 100000000:
			self.steps += 1
		else:
			self.done = True
		return nones, nones, self.done, None

	def reset(self):
		return super().reset()

	def display(self):
		cv2.rectangle(self.img, (0, 0), (self.w, self.h), (255, 255, 255), -1)

		self.hud('Ticks: %f' % self.steps)
		self.hud('Ticks per frame: %f' % self.display_interval)
		self.hud('Scale: %f' % self.scale)
		self.hud('___')
		super().display()
