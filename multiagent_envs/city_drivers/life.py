from typing import List
from typing import TYPE_CHECKING

from multiagent_envs.city_drivers.transport import Intersection
from multiagent_envs.geometry import Point

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
	def __init__(self, pos: Point, vel: Point, mass=.01):
		self.pos, self.vel, self.mass = pos, vel, mass
		self.force = 0
		self.closest = ClosestIntersectionCache()

	def update_closest_cache(self, intersections: List[Intersection], tick):
		if tick == self.closest.tick:
			return self.closest.intersection
		self.closest.distances = [mag(self.pos - intersection) for intersection in intersections]
		closest_intersection_index = int(np.argmin(self.closest.distances))
		self.closest.intersection = intersections[closest_intersection_index]
		self.closest.distance = self.closest.distances[closest_intersection_index]
		return self.closest


class Life:
	def __init__(self, city: 'City'):
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
		self.hotspots.append(Hotspot(spot, Point(0, 0)))

	def dissipate(self):
		for a_i, a in enumerate(self.hotspots):
			a.force *= 0
			for b_i, b in enumerate(self.hotspots):
				if a_i != b_i:
					diff = a.pos - b.pos
					a.force += a.mass * b.mass * diff / mag(diff) ** 3
			a.vel += (a.force + (np.random.random_sample(2) - .5) * self.noise) / a.mass
			a_speed = mag(a.vel)
			a.vel *= max(0, (1 - self.friction - self.drag * a_speed)) * a.mass / (a.mass + self.growth)
			a.mass += self.growth
			a.pos += a.vel / a.mass

	def tick(self):
		self.dissipate()

	def update_closest_intersection_cache(self, intersections: List[Intersection], tick: int = None):
		if tick is None:
			tick = self.city.ticks
		for hotspot in self.hotspots:
			hotspot.update_closest_cache(intersections, tick)

	def most_deprived_hotspot(self, intersections: List[Intersection], tick: int = None) -> Hotspot:
		assert len(self.hotspots) > 0
		inter_list = list(intersections)
		self.update_closest_intersection_cache(inter_list, tick)
		return self.hotspots[int(np.argmax([hotspot.closest.distance for hotspot in self.hotspots]))]
