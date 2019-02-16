import cv2
import numpy as np

from multiagent_envs.const import DIRECTIONS
from multiagent_envs.env import MultiAgentEnv2d
from multiagent_envs.geometry import Point
from multiagent_envs.negotiator import DemandJudge


class AgentAction(object):
	def __init__(self, forward, turn):
		self.forward = np.clip(forward, -1, 1).astype(np.int8)
		self.turn = np.clip(turn, -1, 1).astype(np.int8)


class AgentState(object):
	def __init__(self, pos: Point, orientation: int, goal: 'Goal'):
		self.pos = pos
		self.vel = Point(0, 0)
		self.orientation = orientation
		self.goal = goal
		self.health = 1

	def action_update(self, action: AgentAction):
		self.orientation = (self.orientation + action.turn) % 4
		self.pos += DIRECTIONS[self.orientation] * 10


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


class SWAT(MultiAgentEnv2d, DemandJudge):
	def __init__(self, n_agents: int, goal: Goal, w=1280, h=720):
		super().__init__(n_agents, w, h)
		self.scale = 1

		self.goal = goal

		if goal.noun.adjective is None:
			goal.noun.adjective = self.sample_unique_points(1)[0]

		self.agent_states = [AgentState(point, orientation, goal)
							 for point, orientation in
							 zip(self.sample_unique_points(n_agents), self.sample_orientation(n_agents))]

	def sample_unique_points(self, size):
		return [self.pos_from_seed(seed) for seed in np.random.choice(range(self.w * self.h), size, replace=False)]

	def pos_from_seed(self, seed: int):
		w, h = self.w, self.h
		return Point(seed % w - int(w / 2), int(seed / w) - h / 2)

	def step(self, actions):
		super().step(actions)
		for agent_state, action in zip(self.agent_states, actions):
			agent_state.action_update(action)

		if self.goal.verb.type == Goal.Verb.Type.REACH:
			rews = [-1 for _ in range(self.n_agents)]
			return self.agent_states, rews, self.done, None
		else:
			raise AssertionError('Unknown goal verb type: ' + self.goal.verb.type)

	def reset(self):
		self.steps = 0
		return super().reset()

	def display_agent(self, agent_state: AgentState):
		center = self.window_point(agent_state.pos)
		cv2.circle(self.img, center, int(self.scale * 10),
				   (255 * np.array([agent_state.health, 0, 1 - agent_state.health])).tolist(), -1)
		cv2.line(self.img, center, self.window_point(agent_state.pos + 10 * DIRECTIONS[agent_state.orientation]),
				 (0, 255, 0))

	def display(self):
		cv2.rectangle(self.img, (0, 0), (self.w, self.h), (255, 255, 255), -1)

		for agent_state in self.agent_states:
			self.display_agent(agent_state)

		if self.goal.verb.type == Goal.Verb.Type.REACH:
			cv2.circle(self.img, self.window_point(self.goal.noun.adjective), self.scale * 5, (0, 255, 0), -1)

		self.hud('Ticks: %f' % self.steps)
		self.hud('Ticks per frame: %f' % self.display_interval)
		self.hud('Scale: %f' % self.scale)
		self.hud('___')
		super().display()

	def sample_orientation(self, size: int):
		return np.random.choice(range(4), size)
