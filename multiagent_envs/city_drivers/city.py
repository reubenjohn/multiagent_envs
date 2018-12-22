import cv2
import numpy as np

from multiagent_envs.city_drivers.life import Life, Hotspot
from multiagent_envs.city_drivers.transport import Infrastructure, Transport, Road, Vehicle
from multiagent_envs.const import Y, X
from multiagent_envs.geometry import Edge
from multiagent_envs.ui import Env2d
from multiagent_envs.ui import escape, w, a, s, d, plus, minus, faster, slower, space, enter
from multiagent_envs.util import overlay_transparent, mag


class City(Env2d):
	def __init__(self):
		super().__init__()
		self._fund = 300

		self.infrastructure = Infrastructure(self)
		self.life = Life(self)
		self.transport = Transport(self)

		self.intersection = np.zeros([16, 16, 4])
		cv2.line(self.intersection, (0, 0), (16, 16), (255, 0, 0, 255))
		cv2.line(self.intersection, (0, 16), (16, 0), (0, 0, 255, 255))

		self.car = cv2.imread('pinpoint.png', cv2.IMREAD_UNCHANGED)
		self.car = cv2.resize(self.car, tuple((np.array(self.car.shape[:2]) / 16).astype(np.int)))

	def display_road(self, road: Road):
		self.display_edge(road, [0, 0, 0], len(road.edges) * 2)
		if road.two_way:
			self.display_edge(road, [0, 255, 0], 1)
		else:
			for edge in road.edges:
				self.display_edge(edge, [255, 128, 0], 1)

	def display_hotspot(self, hotspot: Hotspot):
		center = self.window_point(hotspot.pos)
		speed = int(255 * min(mag(hotspot.vel), 10) / 10)
		cv2.circle(self.img, center, int(self.scale * np.sqrt(hotspot.mass)), (255 - speed, 0, speed), -1)
		cv2.line(self.img, center, self.window_point(hotspot.pos + 10 * hotspot.force), (0, 255, 0))

	def display_vehicle(self, vehicle):
		x, y = self.window_point(
			vehicle.local_src + vehicle.pos / vehicle.road.length * (vehicle.local_dst - vehicle.local_src))
		if vehicle.state == Vehicle.CHILLING:
			cv2.circle(self.img, (x, y), 2, (255, 0, 0), -1)
		elif vehicle.state == Vehicle.RACING:
			cv2.circle(self.img, (x, y), 2, (0, 255, 255), -1)
		elif vehicle.state == Vehicle.FREAKING:
			cv2.circle(self.img, (x, y), 8, (0, 0, 255), -1)
		elif vehicle.state == Vehicle.BASKING_IN_GLORY:
			cv2.circle(self.img, (x, y), 8, (0, 255, 0), -1)
		else:
			cv2.circle(self.img, (x, y), 8, (64, 64, 64), -1)

	def display(self, wait=-1):
		cv2.rectangle(self.img, (0, 0), (self.w, self.h), (255, 255, 255), -1)

		for road in self.infrastructure.roads.values():
			self.display_road(road)

		for intersection in self.infrastructure.intersections:
			x, y = self.window_point(intersection)
			# roi = self.img[x - 8:x + 8, y - 8:y + 8]
			overlay_transparent(self.img, self.intersection, x - 8, y - 8)

		for vehicle in self.transport.vehicles:
			self.display_vehicle(vehicle)
		# overlay_transparent(self.img, self.car, x - 8, y - 8)

		for hotspot in self.life.hotspots:
			self.display_hotspot(hotspot)

		for hotspot in self.life.hotspots:
			hp = hotspot.pos
			ci = hotspot.closest.intersection
			self.display_edge(Edge(hp, ci), (0 if hotspot == self.infrastructure.mdh else 255, 0, 255))
			cv2.putText(self.img, str(hotspot.closest.distance), self.window_point((hp + ci) / 2),
						cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))

		self.hud('Ticks: %f' % self.ticks)
		self.hud('Scale: %f' % self.scale)
		self.hud('Fund: %f' % self._fund)
		self.hud('Vehicles: %d' % len(self.transport.vehicles))
		self.hud('Hotspots: %d' % len(self.life.hotspots))
		self.hud('Intersections: %d' % len(self.infrastructure.intersections))
		self.hud('Total hotspot speed: %f' % self.life.total_hotspot_speed)
		self.hud('Vehicle CBR: %f' % self.transport.vehicle_cost_benefit_ratio())
		self.hud('MUR CBR: %f' % self.infrastructure.road_cost_benefit_ratio(self.infrastructure.most_used_road()))
		if self.infrastructure.mdh is not None:
			self.hud('HIC CBR: %f' % self.infrastructure.intersection_cost_benefit_ratio(self.infrastructure.mdh))
		super().display()

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
		if self.ticks % 10 == 0:
			self.fund(10)
		self.transport.tick()
		self.life.tick()
		self.ticks += 1

	def fund(self, fund: int):
		self._fund += fund
		self.life.fund(self._fund)
		self._fund -= self.transport.fund(self._fund)
		self._fund -= self.infrastructure.fund(self._fund)
