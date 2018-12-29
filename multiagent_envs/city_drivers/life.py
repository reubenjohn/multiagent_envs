from typing import List
from typing import TYPE_CHECKING

from multiagent_envs.city_drivers.transport.common import Intersection
from multiagent_envs.city_drivers.transport.driver import Router
from multiagent_envs.city_drivers.transport.infrastructure import Road
from multiagent_envs.geometry import Point
from multiagent_envs.negotiator import DemandNegotiator, RecursiveDemandJudge

if TYPE_CHECKING:
	from multiagent_envs.city_drivers.city import City
from multiagent_envs.util import *


class ClosestCache:
	def __init__(self):
		self.obj = None
		self.tick = -1
		self.separation = -1.
		self.separations = None


class Hotspot:
	cost = 100

	def __init__(self, pos: Point, vel: Point, mass=.01):
		self.pos, self.vel, self.mass = pos, vel, mass
		self.force = 0
		self.closest_intersection = ClosestCache()
		self.least_interconnected = ClosestCache()

	def update_closest_intersection_cache(self, intersections: List[Intersection], tick):
		if tick == self.closest_intersection.tick:
			return self.closest_intersection
		self.closest_intersection.separations = [mag(self.pos - intersection) for intersection in intersections]
		closest_intersection_index = int(np.argmin(self.closest_intersection.separations))
		self.closest_intersection.obj = intersections[closest_intersection_index]
		self.closest_intersection.separation = self.closest_intersection.separations[closest_intersection_index]
		return self.closest_intersection

	def cache_hotspot_connectivity(self, hotspots, router: Router, tick):
		if tick == self.least_interconnected.tick:
			return self.least_interconnected
		self_intersection = self.closest_intersection.obj
		candidates = [(hotspot, hotspot.closest_intersection.obj)
					  for hotspot in hotspots
					  if self_intersection != hotspot.closest_intersection.obj
					  and hotspot != self]
		if len(candidates) == 0:
			return None
		heaviest_hotspot = max(hotspot.mass for hotspot in hotspots)
		self.least_interconnected.separations = [
			(1 - (mag(self_intersection - intersection) / router[self_intersection][intersection].dist)) * (
					(self.mass + hotspot.mass) / (2 * heaviest_hotspot)
			)
			for hotspot, intersection in candidates
		]
		least_separated_index = int(np.argmax(self.least_interconnected.separations))
		self.least_interconnected.obj = candidates[least_separated_index][0]
		assert self.least_interconnected.obj != self and self.least_interconnected.obj.closest_intersection.obj != self_intersection
		self.least_interconnected.separation = self.least_interconnected.separations[least_separated_index]
		return self.least_interconnected


class HotspotAccessibilityNegotiator(DemandNegotiator):
	def __init__(self, life: 'Life') -> None:
		super().__init__('Hotspot Demand', Hotspot.cost)
		self.life = life

	def setup(self, scenario):
		return self.life.total_hotspot_speed() + .01

	def compute_demand(self, scenario, total_hotspot_speed):
		return min(1 / total_hotspot_speed, 1)

	def fund(self, scenario, fund: float, setup):
		self.life.add_hotspot()
		return Hotspot.cost


class InterHotspotConnectivityNegotiator(DemandNegotiator):
	def __init__(self, life: 'Life') -> None:
		super().__init__('Hotspot Connectivity Demand', Road.min_length)
		self.life = life
		self.src = self.dst = None  # type: Intersection

	def setup(self, scenario):
		return self.life.least_interconnected_hotspot(scenario)

	def compute_demand(self, scenario, lih: 'Hotspot'):
		if lih is not None and lih.least_interconnected.separation != 0:
			return lih.least_interconnected.separation
		return 0

	def fund(self, scenario, fund: float, lih: 'Hotspot'):
		src, dst = lih.closest_intersection.obj, lih.least_interconnected.obj.closest_intersection.obj
		self.src, self.dst = lih.closest_intersection.obj, lih.least_interconnected.obj.closest_intersection.obj
		diff = (dst - src)
		diff_mag = mag(diff)
		if fund >= diff_mag:
			self.life.city.infrastructure.connect_intersections(src, dst, 1, True)
			return diff_mag
		else:
			return 0
			self.life.city.infrastructure.extend_road(src, src + unit(diff) * fund, 1, True)
			return fund


class Life(RecursiveDemandJudge):
	def __init__(self, city: 'City'):
		super().__init__('Life Demand')
		self.friction = 0.0
		self.drag = 0.001
		self.noise = 0.01
		self.growth = 0.01
		self.hotspots = []  # type: List[Hotspot]
		self.momentum = np.ones([len(self.hotspots), len(self.hotspots)], np.float16)
		self.city = city

		self.add_negotiator(HotspotAccessibilityNegotiator(self)) \
			.add_negotiator(InterHotspotConnectivityNegotiator(self))

	def add_hotspot(self, spot: Point = None):
		if spot is None:
			spot = Point(self.city.infrastructure.most_used_road().a)
		hotspot = Hotspot(spot, Point(0, 0))
		self.hotspots.append(hotspot)
		self.cache_closest_intersection()
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

	def cache_closest_intersection(self, intersections: List[Intersection] = None, tick: int = None):
		if tick is None:
			tick = self.city.ticks
		if intersections is None:
			intersections = list(self.city.infrastructure.intersections)
		for hotspot in self.hotspots:
			hotspot.update_closest_intersection_cache(intersections, tick)

	def cache_hotspot_connectivity(self, tick: int = None):
		if tick is None:
			tick = self.city.ticks
		for hotspot in self.hotspots:
			hotspot.cache_hotspot_connectivity(self.hotspots, self.city.infrastructure.router, tick)

	def most_deprived_hotspot(self, intersections: List[Intersection], tick: int = None) -> Hotspot:
		assert len(self.hotspots) > 0
		self.cache_closest_intersection(list(intersections), tick)
		return self.hotspots[int(np.argmax([hotspot.closest_intersection.separation for hotspot in self.hotspots]))]

	def least_interconnected_hotspot(self, tick: int = None) -> Hotspot:
		if len(self.hotspots) > 1:
			self.cache_closest_intersection(None, tick)
			self.cache_hotspot_connectivity(tick)
			return self.hotspots[int(np.argmax([hotspot.least_interconnected.separation for hotspot in self.hotspots]))]

	def total_hotspot_speed(self):
		return sum(mag(hotspot.vel) for hotspot in self.hotspots)
