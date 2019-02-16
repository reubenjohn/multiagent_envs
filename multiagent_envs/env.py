from threading import Thread
from time import sleep

from multiagent_envs.ui import *


class MultiAgentEnv2d(Window):
	def __init__(self, n_agents: int, w=1280, h=720):
		self.n_agents = n_agents
		super().__init__(w, h)

		self.is_open = True
		self.input_thread = Thread(target=self._input_thread)
		self.display_ticks_to_skip = 0

		self.steps = 1
		self.display_interval = 1
		self.paused = False
		self.n_step_over = 0

		self.done = False

	def reset(self):
		self.done = False
		return [None for _ in range(self.n_agents)]

	def step(self, actions):
		assert not self.done
		assert len(actions) == self.n_agents
		while self.paused and self.n_step_over == 0:
			sleep(.1)

	def handle_input(self, key: int):
		if key == escape:
			self.is_open = False
		elif key in {w, a, s, d}:
			self.focus += 10 * {w: Y, a: -X, s: -Y, d: X}[key] / self.scale * 4
		elif key in {plus, minus}:
			self.scale = self.scale * {plus: 1.4, minus: 0.6}[key]
		elif key in {faster, slower}:
			self.display_interval = max(self.display_interval * {faster: 1.5, slower: 0.5}[key], 1)
		elif key == space:
			self.paused = not self.paused
		elif key == enter:
			if self.paused:
				self.n_step_over = 1

	def close(self):
		self.is_open = False
		self.input_thread.join()
		cv2.destroyAllWindows()

	def open(self):
		self.input_thread.start()

	def _input_thread(self):
		while self.is_open:
			if self.display_ticks_to_skip > 0:
				self.display_ticks_to_skip -= 1
			else:
				self.display_ticks_to_skip = self.display_interval
				self.display()

			key = cv2.waitKey(0 if self.paused else 1)
			self.handle_input(key)
