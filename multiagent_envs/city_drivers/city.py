import cv2
import numpy as np

from multiagent_envs.city_drivers.life import Life, Hotspot, InterHotspotConnectivityNegotiator
from multiagent_envs.city_drivers.transport.infrastructure import Infrastructure, Road, Vehicle
from multiagent_envs.city_drivers.transport.transport import Transport
from multiagent_envs.geometry import Edge
from multiagent_envs.negotiator import DemandJudge, RecursiveDemandJudge
from multiagent_envs.ui import Env2d
from multiagent_envs.util import overlay_transparent, mag


class City(Env2d, DemandJudge):
	def __init__(self):
		super().__init__()
		self.display_interval = 1
		self._fund = 300

		self.infrastructure = Infrastructure(self)
		self.life = Life(self)
		self.transport = Transport(self)

		self.add_negotiator(self.infrastructure) \
			.add_negotiator(self.transport) \
			.add_negotiator(self.life)

		self.intersection = np.zeros([16, 16, 4])
		cv2.line(self.intersection, (0, 0), (16, 16), (255, 0, 0, 255))
		cv2.line(self.intersection, (0, 16), (16, 0), (0, 0, 255, 255))

		self.car = cv2.imread('pinpoint.png', cv2.IMREAD_UNCHANGED)
		self.car = cv2.resize(self.car, tuple((np.array(self.car.shape[:2]) / 16).astype(np.int)))

	def demand_hud(self, negotiator_name: str, demand: float, min_demand: float, max_demand: float):
		relative_demand = (demand - min_demand) / (max_demand - min_demand)
		color = (.6 - .6 * float(relative_demand), 1 - 1 * float(relative_demand), .6 + .4 * float(relative_demand))
		self.hud('%s: ' % negotiator_name, ('%.4f' % demand, color))

	def display_road(self, road: Road):
		self.display_edge(road, [0, 0, 0], len(road.edges) * 2)
		if road.two_way:
			self.display_edge(road, [0, 255, 0], 1)
		else:
			for edge in road.edges:
				self.display_edge(edge, [255, 128, 0], 1)

	def display_hotspot(self, hotspot: Hotspot):
		hp = hotspot.pos

		center = self.window_point(hp)
		speed = int(255 * min(mag(hotspot.vel), 10) / 10)
		cv2.circle(self.img, center, int(self.scale * np.sqrt(hotspot.mass)), (255 - speed, 0, speed), -1)
		cv2.line(self.img, center, self.window_point(hp + 10 * hotspot.force), (0, 255, 0))

		ci = hotspot.closest_intersection.obj
		# cv2.putText(self.img, str(hotspot.closest_intersection.separation), self.window_point((hp + ci) / 2),
		# 			cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
		self.display_edge(Edge(hp, ci), (255, 0, 255))

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
			cv2.circle(self.img, (x, y), 16, (0, 255, 0), -1)
		else:
			cv2.circle(self.img, (x, y), 8, (64, 64, 64), -1)

		goal = self.window_point(vehicle.global_dst)
		cv2.circle(self.img, goal, 4, (0, 255, 0), -1)

	def display_demands(self, judge: DemandJudge):
		zeroed_demands = [demand if demand is not None else 0 for demand in judge.demands]
		max_demand, min_demand = max(zeroed_demands), min(zeroed_demands)
		max_demand = 1 if max_demand == 0 else max_demand
		for negotiator, demand in zip(judge.negotiators, judge.demands):
			if demand is not None:
				if isinstance(negotiator, InterHotspotConnectivityNegotiator):
					if negotiator.src is not None:
						self.display_edge(Edge(negotiator.src, negotiator.dst), (0, 255, 255), 2)
					self.demand_hud(negotiator.name, demand, min_demand, max_demand)
				elif not isinstance(negotiator, RecursiveDemandJudge):
					self.demand_hud(negotiator.name, demand, min_demand, max_demand)
				elif isinstance(negotiator, DemandJudge):
					self.display_demands(negotiator)
				else:
					raise RuntimeError('Unknown negotiator type' + type(negotiator).__name__)

	def display(self, wait=-1):
		cv2.rectangle(self.img, (0, 0), (self.w, self.h), (255, 255, 255), -1)

		for road in self.infrastructure.roads.values():
			self.display_road(road)

		for intersection in self.infrastructure.intersections:
			x, y = self.window_point(intersection)
			overlay_transparent(self.img, self.intersection, x - 8, y - 8)

		for vehicle in self.transport.vehicles:
			self.display_vehicle(vehicle)

		for hotspot in self.life.hotspots:
			self.display_hotspot(hotspot)

		self.hud('Ticks: %f' % self.ticks)
		self.hud('Ticks per frame: %f' % self.display_interval)
		self.hud('Scale: %f' % self.scale)
		self.hud('Fund: %f' % self._fund)
		self.hud('Vehicles: %d' % len(self.transport.vehicles))
		self.hud('Hotspots: %d' % len(self.life.hotspots))
		self.hud('Intersections: %d' % len(self.infrastructure.intersections))
		self.hud('___')
		self.display_demands(self)
		super().display()

	def handle_input(self, key: int):
		if super().handle_input(key):
			return True
		if key != -1:
			print(key)

	def tick(self):
		if self.ticks % 10 == 0:
			self.fund(10)
		self.transport.tick()
		self.life.tick()
		self.ticks += 1

	def fund(self, fund: int):
		self._fund += fund
		self._fund -= self.fund_best_negotiator(self.ticks, self._fund)
