from typing import List

import cv2
import numpy as np

from multiagent_envs.common.common import neighbours_generator
from multiagent_envs.env import MultiAgentEnv2d, Point
from multiagent_envs.util import mag


class AgentAction(object):
	def __init__(self, acc):
		self.acc = acc


class AgentOb(object):
	def __init__(self, vel, behind_dist, behind_vel, forward_dist, forward_vel):
		self.vel = vel
		self.behind_dist = behind_dist
		self.behind_vel = behind_vel
		self.forward_dist = forward_dist
		self.forward_vel = forward_vel


class AgentState(object):
	def __init__(self):
		self.pos = None
		self.vel = None
		self.disorientation = None

	def action_update(self, action: AgentAction, prev: 'AgentState', nxt: 'AgentState', env: 'RoundAbout'):
		if self.disorientation > 0:
			self.disorientation -= 1
			self.vel = 0
			return
		acc = np.clip(action.acc, -env.max_acc, env.max_acc)

		free_space_in_front = max(0, (nxt.pos - self.pos) % 1 - env.car_arc_ratio)

		self.vel = max(0, (self.vel + acc))

		displacement = min(self.vel, free_space_in_front)

		if displacement == free_space_in_front:
			self.disorientation = env.disorientation

		self.pos = (self.pos + displacement) % 1

	def reset(self, pos: float):
		self.pos = pos
		self.vel = 0
		self.disorientation = 0
		return self.pos, self.vel

	def angular_pos(self, offset_arc_ratio: [float, Point] = .0) -> [float, Point]:
		return (self.pos + offset_arc_ratio) * 2 * np.pi + np.pi / 2


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


class Agent:
	def __init__(self, state: AgentState, debug: AgentDebug):
		self.state = state
		self.debug = debug

	def reset(self, pos):
		return self.state.reset(pos)


class RoundAbout(MultiAgentEnv2d):
	OVERSPEED_COLLISION_RATIO = 3

	def __init__(self, n_agents: int, occupancy_ratio: float = .5, disorientation=50, window_width: int = 720):
		super().__init__(n_agents, window_width, window_width)

		self.scale = window_width / 3
		self.focus = Point(.4, 0)

		self.disorientation = disorientation
		self.max_steps = 100 * n_agents
		self.car_arc_ratio = occupancy_ratio / n_agents
		self.max_vel = (1 - occupancy_ratio) * 2 / ((2 + disorientation) * n_agents)
		self.max_acc = (1 - occupancy_ratio) * 2 / (
				(2 + disorientation) * n_agents) ** 2 * RoundAbout.OVERSPEED_COLLISION_RATIO
		self.n_slots = int(1 / self.car_arc_ratio)

		self.agents = [Agent(AgentState(), AgentDebug()) for _ in range(self.n_agents)]
		self.agent_states = [agent.state for agent in self.agents]
		self.agent_debugs = [agent.debug for agent in self.agents]

	def sample_unique_positions(self, size):
		return np.sort([slot / self.n_slots for slot in np.random.choice(range(self.n_slots), size, replace=False)])

	def step(self, actions):
		super().step(actions)
		if self.done:
			raise AssertionError('Episode has already terminated!')

		rews = [agent.state.vel for agent in self.agents]
		for (prev, curr, nxt), action in zip(neighbours_generator(self.agents), actions):
			curr.state.action_update(action, prev.state, nxt.state, self)

		self.done = self.steps == self.max_steps
		return self.observations(), rews, self.done, self.agent_debugs

	def reset(self):
		self.steps = 0
		super().reset()

		[agent.reset(pos) for agent, pos in zip(self.agents, self.sample_unique_positions(self.n_agents))]
		return self.observations()

	def relative_roundabout_pos(self, angular_pos):
		return Point(np.cos(angular_pos), np.sin(angular_pos))

	def display_agent(self, index, agent_state: AgentState, agent_debug: AgentDebug):
		fr_point = self.window_transform(self.relative_roundabout_pos(agent_state.angular_pos(-self.car_arc_ratio / 2)))
		bk_point = self.window_transform(self.relative_roundabout_pos(agent_state.angular_pos(self.car_arc_ratio / 2)))
		cv2.line(self.img, fr_point, bk_point, agent_debug.color, 3)
		cv2.putText(self.img,
					'[%d] %.2fm @ %.1f%% %s' % (
						index, agent_state.pos, 100 * agent_state.vel / self.max_vel,
						'?!' if agent_state.disorientation > 0 else ''
					),
					fr_point, cv2.FONT_HERSHEY_SIMPLEX, self.window_transform_float(.002),
					(0, 0, 255) if agent_state.disorientation > 0 else agent_debug.color
					)

	def display(self, show_debug_hud: bool = True):
		cv2.rectangle(self.img, (0, 0), (self.w, self.h), (255, 255, 255), -1)
		center, radius, thickness = self.window_point(Point(0, 0)), self.window_transform(1), self.window_transform(.1)
		cv2.circle(self.img, center, radius, (255, 255, 255), thickness)

		for index, (state, agent_debug) in enumerate(zip(self.agent_states, self.agent_debugs)):
			self.display_agent(index, state, agent_debug)

		self.hud('Avg. speed: %.3f' % (sum([state.vel / self.max_vel for state in self.agent_states]) / self.n_agents))
		super().display(show_debug_hud)

	def observations(self) -> List[AgentOb]:
		return [
			AgentOb(curr.vel, (curr.pos - prev.pos) % 1, prev.vel, (nxt.pos - curr.pos) % 1, nxt.vel)
			for prev, curr, nxt in neighbours_generator(self.agent_states)
		]
