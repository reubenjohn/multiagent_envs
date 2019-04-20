from multiagent_envs.ui import *


class MultiAgentEnv2d(Window):
	def __init__(self, n_agents: int, w=1280, h=720):
		self.n_agents = n_agents
		super().__init__(w, h)

		self.is_open = True

		self.steps = 1
		self.display_interval = 1
		self.paused = False
		self.n_step_over = 0

		self.done = False

	def reset(self):
		self.done = False
		self.steps = 0

	def step(self, actions):
		assert not self.done
		assert len(actions) == self.n_agents
		self.steps += 1

	def handle_input(self, key: int = None):
		if key is None:
			key = cv2.waitKey(0 if self.paused else 1)
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
		print('Destroying all windows')
		cv2.destroyAllWindows()

	def open(self):
		pass

	def display(self, show_debug_hud: bool = False):
		if show_debug_hud:
			self.hud('___')
			self.hud('Ticks: %d' % self.steps)
			self.hud('Ticks per frame: %.1f' % self.display_interval)
			self.hud('Scale: %.2f' % self.scale)
		super().display()
