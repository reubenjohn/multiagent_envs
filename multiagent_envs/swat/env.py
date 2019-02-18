from math import ceil
from typing import Tuple, List

import cv2
import numpy as np

from multiagent_envs.const import DIRECTIONS
from multiagent_envs.env import MultiAgentEnv2d
from multiagent_envs.geometry import Point
from multiagent_envs.negotiator import DemandJudge
from multiagent_envs.util import mag


class AgentAction(object):
	def __init__(self, direction, turn):
		self.direction = direction
		self.turn = np.clip(turn, -1, 1).astype(np.int8)


class AgentState(object):
	def __init__(self):
		self.initial_pos = self.pos = None
		self.direction = None
		self.goal = None
		self.health = None

	@staticmethod
	def normalize_direction(orientation: int) -> float:
		return (orientation - 2) / 2

	@property
	def denormalized_direction(self) -> int:
		return int(self.direction * 2) + 2

	def action_update(self, action: AgentAction, env: 'SWAT'):
		self.direction = self.normalize_direction(self.denormalized_direction + int(np.clip(action.turn, -1, 1)) % 4)
		c_size = env.cache.corner_offset[0]
		self.pos += DIRECTIONS[action.direction] * 2 * env.cache.corner_offset
		self.pos.clip([-1 + c_size, -1 + c_size], [1 - c_size, 1 - c_size], self.pos)

	def reset(self, pos: Point, orientation: int, goal: 'Goal'):
		self.initial_pos = pos
		self.pos = Point(pos)
		self.direction = AgentState.normalize_direction(orientation)
		self.goal = goal
		self.health = None


class Goal:
	class Verb:
		class Type:
			REACH = 'reach'

		def __init__(self, verb_type: str, adverb):
			self.type = verb_type
			self.adverb = adverb

	class Noun:
		class Type:
			POINT = 'point'

		def __init__(self, noun_type: str, adjective):
			self.type = noun_type
			self.adjective = adjective

	def __init__(self, verb: Verb, noun: Noun):
		self.verb, self.noun = verb, noun


class Cache:
	def __init__(self):
		self.diagonal = None
		self.normalized_diagonal = np.sqrt(8)
		self.cell_size = None
		self.corner_offset = None

	def update(self, window_width, tiles_shape):
		self.diagonal = mag(tiles_shape)
		self.cell_size = window_width / tiles_shape[0]
		self.corner_offset = np.array([1 / tiles_shape[0], 1 / tiles_shape[1]])


class AgentDebug:
	def __init__(self):
		self.color = (255, 0, 0)


class SWAT(MultiAgentEnv2d, DemandJudge):
	def __init__(self, n_agents: int, goal: Goal, tiles_shape: Tuple[int, int] = (10, 10),
				 window_width: int = 720):
		window_height = ceil(window_width * tiles_shape[1] / tiles_shape[0])
		super().__init__(n_agents, window_width, window_height)
		self.scale = window_width / 2
		self.tiles_shape = np.array(tiles_shape)
		self.cache = Cache()
		self.cache.update(window_width, self.tiles_shape)

		self.goal = goal

		self.agent_states = [AgentState() for _ in range(self.n_agents)]
		self.agent_debugs = [AgentDebug() for _ in range(self.n_agents)]
		self.reset()

	def sample_unique_points(self, size):
		return [self.pos_from_seed(seed) for seed in
				np.random.choice(range(self.tiles_shape[0] * self.tiles_shape[1]), size, replace=False)]

	def pos_from_seed(self, seed: int):
		n_x, n_y = self.tiles_shape
		return (Point(seed % n_x, int(seed / n_x)) - self.tiles_shape / 2) / (
				self.tiles_shape / 2) + self.cache.corner_offset

	def step(self, actions):
		super().step(actions)
		for agent_state, action in zip(self.agent_states, actions):
			agent_state.action_update(action, self)

		if self.steps == self.tiles_shape[0] + self.tiles_shape[1]:
			self.done = True
			return None, None, self.done, self.agent_debugs

		if self.goal.verb.type == Goal.Verb.Type.REACH:
			rews = [(mag(state.initial_pos - state.goal.noun.adjective) - mag(
				state.pos - state.goal.noun.adjective)) / self.cache.normalized_diagonal for state in self.agent_states]
			for state, rew in zip(self.agent_states, rews):
				state.health = rew
			return self.agent_states, rews, self.done, self.agent_debugs
		else:
			raise AssertionError('Unknown goal verb type: ' + self.goal.verb.type)

	def reset(self) -> List[AgentState]:
		if self.agent_states[0].health is not None:
			healths = [state.health for state in self.agent_states]
			print("%.4f, %.4f" % (max(healths), min(healths)))
		self.steps = 0
		super().reset()

		self.goal.noun.adjective = self.sample_unique_points(1)[0]

		[state.reset(point, orientation, self.goal)
		 for state, point, orientation in
		 zip(self.agent_states,
			 self.sample_unique_points(self.n_agents), self.sample_orientation(self.n_agents))]
		if self.goal.verb.type == Goal.Verb.Type.REACH:
			return self.agent_states

	def display_agent(self, agent_state: AgentState, agent_debug: AgentDebug):
		center = self.window_point(agent_state.pos)
		cv2.circle(self.img, center, int(self.cache.cell_size / 2), agent_debug.color, -1)
		cv2.rectangle(self.img, self.window_point(agent_state.pos - self.cache.corner_offset / 2),
					  self.window_point(agent_state.pos + self.cache.corner_offset / 2),
					  agent_debug.youth_color, -1)
		cv2.line(self.img, center,
				 self.window_point(agent_state.pos + self.cache.corner_offset * DIRECTIONS[agent_state.direction]),
				 agent_debug.gene_novelty_color, 4)

	def display(self):
		cv2.rectangle(self.img, (0, 0), (self.w, self.h), (255, 255, 255), -1)

		for state in self.agent_states:
			cv2.line(self.img, self.window_point(state.pos), self.window_point(state.initial_pos), (1, 0, 1))
		for state, agent_debug in zip(self.agent_states, self.agent_debugs):
			self.display_agent(state, agent_debug)

		if self.goal.verb.type == Goal.Verb.Type.REACH:
			center, corner_offset = self.goal.noun.adjective, self.cache.corner_offset
			cv2.rectangle(self.img, self.window_point(center - corner_offset),
						  self.window_point(center + corner_offset), (0, 0, 0), -1)

		self.hud('Ticks: %d' % self.steps)
		self.hud('Ticks per frame: %.1f' % self.display_interval)
		self.hud('Scale: %.2f' % self.scale)
		self.hud('___')
		super().display()

	def sample_orientation(self, size: int):
		return np.random.choice(range(4), size)
