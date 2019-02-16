from typing import Union

import cv2

from multiagent_envs.const import *
from multiagent_envs.geometry import Edge, Point

w, a, s, d = 119, 97, 115, 100
plus, minus = 43, 95
faster, slower = 93, 91
space = 32
enter = 13
escape = 27


class Window:
	def __init__(self, w=1280, h=720):
		self.w = w
		self.h = h
		self.img = self.reset_window()
		self.frame_hud_count = 0

		self.focus = Point(0, 0)
		self.scale = 5
		super().__init__()

	def reset_window(self):
		self.img = np.ones([self.h, self.w, 3])
		return self.img

	def window_transform(self, obj: Union[Point, Edge]):
		if isinstance(obj, Point):
			return Point(self.w / 2 + (obj[0] - self.focus[0]) * self.scale,
						 self.h / 2 - (obj[1] - self.focus[1]) * self.scale)
		elif isinstance(obj, Edge):
			return Edge(self.window_transform(obj.a), self.window_transform(obj.b))
		else:
			raise AssertionError('Object ' + str(obj) + ' must be an instance of either Point or Edge')

	def window_point(self, p: Point):
		return tuple(self.window_transform(p).astype(np.int))

	def display_edge(self, e: Edge, color=None, thickness=1):
		if color is None:
			color = [128, 128, 128]
		e = self.window_transform(e)
		cv2.line(self.img, tuple(e.a.astype(np.int)), tuple(e.b.astype(np.int)), color, thickness)

	def hud(self, *parts):
		self.frame_hud_count += 1
		accumulator = 0
		for part in parts:
			if isinstance(part, tuple):
				part, color = part
			else:
				color = (0, 0, 0)
			cv2.putText(self.img, part, (accumulator, self.frame_hud_count * 20), cv2.FONT_HERSHEY_PLAIN, 1, color)
			accumulator += 9 * len(part)

	def display(self):
		cv2.imshow('image', self.img)
		self.frame_hud_count = 0


class Env2d(Window):
	def __init__(self, w=1280, h=720):
		super().__init__(w, h)
		self.ticks = 1
		self.display_interval = 1
		self.paused = False

	def tick(self):
		raise NotImplementedError

	def handle_input(self, key: int):
		if key == escape:
			return True
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
				self.tick()

	def iterate(self):
		if not self.paused:
			self.tick()
		if self.ticks % int(self.display_interval) == 0:
			self.display()
			key = cv2.waitKey(0 if self.paused else 1)
			return self.handle_input(key)

	def run(self):
		while True:
			if self.iterate():
				break
		cv2.destroyAllWindows()
