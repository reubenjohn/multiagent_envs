from typing import List
from typing import TYPE_CHECKING

from multiagent_envs.city_drivers.transport.common import Intersection
from multiagent_envs.geometry import Point
from multiagent_envs.negotiator import DemandNegotiator

if TYPE_CHECKING:
	from multiagent_envs.city_drivers.city import City
from multiagent_envs.util import *


class ClosestIntersectionCache:
	def __init__(self):
		self.intersection = None
		self.tick = -1
		self.distance = -1
		self.distances = None


class Hotspot:
	cost = 100

	def __init__(self, pos: Point, vel: Point, mass=.01):
		self.pos, self.vel, self.mass = pos, vel, mass
		self.force = 0
		self.closest = ClosestIntersectionCache()

	def update_closest_cache(self, intersections: List[Intersection], tick):
		if tick == self.closest.tick:
			return self.closest
		self.closest.distances = [mag(self.pos - intersection) for intersection in intersections]
		closest_intersection_index = int(np.argmin(self.closest.distances))
		self.closest.intersection = intersections[closest_intersection_index]
		self.closest.distance = self.closest.distances[closest_intersection_index]
		return self.closest


class Life(DemandNegotiator):
	def __init__(self, city: 'City'):
		super().__init__('Hotspot Demand', Hotspot.cost)
		self.friction = 0.0
		self.drag = 0.001
		self.noise = 0.01
		self.growth = 0.01
		self.hotspots = []  # type: List[Hotspot]
		self.momentum = np.ones([len(self.hotspots), len(self.hotspots)], np.float16)
		self.city = city

	def add_hotspot(self, spot: Point = None):
		if spot is None:
			spot = Point(self.city.infrastructure.most_used_road().a)
		hotspot = Hotspot(spot, Point(0, 0))
		self.hotspots.append(hotspot)
		self.update_closest_intersection_cache()
		self.dissipate([hotspot])

	def dissipate(self, hotspots: List[Hotspot] = None):
		hotspots = hotspots or self.hotspots
		for a_i, a in enumerate(hotspots):
			a.force *= 0
			for b_i, b in enumerate(self.hotspots):
				if a_i != b_i:
					diff = a.pos - b.pos
					mag_diff = mag(diff)
					if mag_diff > 0.01:
						a.force += a.mass * b.mass * diff / mag(diff) ** 3
			a.vel += (a.force + (np.random.random_sample(2) - .5) * self.noise) / a.mass
			a_speed = mag(a.vel)
			a.vel *= max(0, (1 - self.friction - self.drag * a_speed)) * a.mass / (a.mass + self.growth)
			a.mass += self.growth
			a.pos += a.vel / a.mass

	def tick(self):
		self.dissipate()

	def update_closest_intersection_cache(self, intersections: List[Intersection] = None, tick: int = None):
		if tick is None:
			tick = self.city.ticks
		if intersections is None:
			intersections = list(self.city.infrastructure.intersections)
		for hotspot in self.hotspots:
			hotspot.update_closest_cache(intersections, tick)

	def most_deprived_hotspot(self, intersections: List[Intersection], tick: int = None) -> Hotspot:
		assert len(self.hotspots) > 0
		self.update_closest_intersection_cache(list(intersections), tick)
		return self.hotspots[int(np.argmax([hotspot.closest.distance for hotspot in self.hotspots]))]

	def total_hotspot_speed(self):
		return sum(mag(hotspot.vel) for hotspot in self.hotspots)

	def setup(self, scenario):
		return self.total_hotspot_speed() + .01

	def compute_demand(self, scenario, total_hotspot_speed):
		return min(1 / total_hotspot_speed, 1)

	def fund(self, scenario, fund: float, setup):
		self.add_hotspot()
		return Hotspot.cost
