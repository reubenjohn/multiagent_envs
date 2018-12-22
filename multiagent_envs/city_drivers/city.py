from typing import Union

import cv2
import numpy as np

from multiagent_envs.city_drivers.life import Life, Hotspot
from multiagent_envs.city_drivers.transport import Infrastructure, Transport, Edge, Road, Vehicle
from multiagent_envs.const import Y, X
from multiagent_envs.ui import Window
from multiagent_envs.ui import escape, w, a, s, d, plus, minus, faster, slower, space, enter
from multiagent_envs.util import Point, overlay_transparent, mag


class City(Window):
	def __init__(self):
		super().__init__()
		self.display_interval = 1
		self.ticks = 1
		self._fund = 300
		self.focus = Point(0, 0)
		self.scale = 5
		self.paused = False

		self.infrastructure = Infrastructure(self)
		self.life = Life(self)
		self.transport = Transport(self)

		self.intersection = np.zeros([16, 16, 4])
		cv2.line(self.intersection, (0, 0), (16, 16), (255, 0, 0, 255))
		cv2.line(self.intersection, (0, 16), (16, 0), (0, 0, 255, 255))

		self.car = cv2.imread('pinpoint.png', cv2.IMREAD_UNCHANGED)
		self.car = cv2.resize(self.car, tuple((np.array(self.car.shape[:2]) / 16).astype(np.int)))

	def camera_transform(self, obj: Union[Point, Edge]):
		if isinstance(obj, Point):
			return Point(self.w / 2 + (obj[0] - self.focus[0]) * self.scale,
						 self.h / 2 - (obj[1] - self.focus[1]) * self.scale)
		elif isinstance(obj, Edge):
			return Edge(self.camera_transform(obj.a), self.camera_transform(obj.b))

	def cv2_point(self, p: Point):
		return tuple(self.camera_transform(p).astype(np.int))

	def display_edge(self, e: Edge, color=None, thickness=1):
		if color is None:
			color = [128, 128, 128]
		e = self.camera_transform(e)
		cv2.line(self.img, (int(e.a[0]), int(e.a[1])), (int(e.b[0]), int(e.b[1])), color, thickness)

	def display_road(self, road: Road):
		self.display_edge(road, [0, 0, 0], len(road.edges) * 2)
		if road.two_way:
			self.display_edge(road, [0, 255, 0], 1)
		else:
			for edge in road.edges:
				self.display_edge(edge, [255, 128, 0], 1)

	def display(self, wait=-1):
		cv2.rectangle(self.img, (0, 0), (self.w, self.h), (255, 255, 255), -1)
		self.hud('Ticks: %f' % self.ticks)
		self.hud('Scale: %f' % self.scale)
		self.hud('Fund: %f' % self._fund)
		self.hud('Vehicles: %d' % len(self.transport.vehicles))
		self.hud('Hotspots: %d' % len(self.life.hotspots))
		self.hud('Intersections: %d' % len(self.infrastructure.intersections))
		self.hud('MUR CBR: %f' % self.infrastructure.road_cost_benefit_ratio(self.infrastructure.most_used_road()))
		if self.infrastructure.mdh is not None:
			self.hud('HIC CBR: %f' % self.infrastructure.intersection_cost_benefit_ratio(self.infrastructure.mdh))
		# self.hud('Vehicles: ' + str([(vehicle.pos, vehicle.vel) for vehicle in self.transport.vehicles]))

		for road in self.infrastructure.roads.values():
			self.display_road(road)

		for intersection in self.infrastructure.intersections:
			x, y = self.cv2_point(intersection)
			# roi = self.img[x - 8:x + 8, y - 8:y + 8]
			overlay_transparent(self.img, self.intersection, x - 8, y - 8)

		for vehicle in self.transport.vehicles:
			x, y = self.cv2_point(
				vehicle.local_src + vehicle.pos / vehicle.road.length * (vehicle.local_dst - vehicle.local_src))
			# roi = self.img[x - 8:x + 8, y - 8:y + 8]
			if vehicle.state == Vehicle.CHILLING:
				cv2.circle(self.img, (x, y), 2, (255, 0, 0), -1)
			elif vehicle.state == Vehicle.RACING:
				cv2.circle(self.img, (x, y), 2, (0, 255, 255), -1)
			elif vehicle.state == Vehicle.BASKING_IN_GLORY:
				cv2.circle(self.img, (x, y), 8, (0, 255, 0), -1)
		# overlay_transparent(self.img, self.car, x - 8, y - 8)

		for hotspot in self.life.hotspots:
			self.display_hotspot(hotspot)

		for hotspot in self.life.hotspots:
			hp = hotspot.pos
			ci = hotspot.closest.intersection
			self.display_edge(Edge(hp, ci), (0 if hotspot == self.infrastructure.mdh else 255, 0, 255))
			cv2.putText(self.img, str(hotspot.closest.distance), self.cv2_point((hp + ci) / 2),
						cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))

		super().display()

	def display_hotspot(self, hotspot: Hotspot):
		center = self.cv2_point(hotspot.pos)
		speed = int(255 * min(mag(hotspot.vel), 10) / 10)
		cv2.circle(self.img, center, int(self.scale * np.sqrt(hotspot.mass)), (255 - speed, 0, speed), -1)
		cv2.line(self.img, center, self.cv2_point(hotspot.pos + 10 * hotspot.force), (0, 255, 0))

	def handle_input(self):
		key = cv2.waitKey(0 if self.paused else 1)
		if key == escape:
			return True
		elif key in {w, a, s, d}:
			self.focus += 10 * {w: Y, a: -X, s: -Y, d: X}[key] / self.scale
		elif key in {plus, minus}:
			self.scale = self.scale * {plus: 1.2, minus: 0.8}[key]
		elif key in {faster, slower}:
			self.display_interval = max(self.display_interval + {faster: 1, slower: -1}[key], 1)
		elif key == space:
			self.paused = not self.paused
		elif key == enter:
			if self.paused:
				self.tick()
		elif key != -1:
			print(key)

	def tick(self):
		self.ticks += 1
		if self.ticks % 10 == 0:
			if self.infrastructure.mdh is None or self.infrastructure.intersection_cost_benefit_ratio(
					self.infrastructure.mdh) > .75:
				self.life.add_hotspot()
			self.fund(10)
		self.transport.tick()
		self.life.tick()

	def fund(self, fund: int):
		self._fund += fund
		self._fund -= self.transport.fund(self._fund)
		self._fund -= self.infrastructure.fund(self._fund)

	def handle_display(self):
		self.display()

	def iterate(self):
		if not self.paused:
			self.tick()
		if self.ticks % self.display_interval == 0:
			self.handle_display()
		return self.handle_input()

	def run(self):
		while True:
			if self.iterate():
				break
		cv2.destroyAllWindows()
