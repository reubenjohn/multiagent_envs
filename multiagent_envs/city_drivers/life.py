import random
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
	mass_multiplier = 1.5
	mature_mass = 5
	cost = 100

	def __init__(self, pos: Point, vel: Point, mass=10):
		self.pos, self.vel, self.mass = pos, vel, mass
		self.force = 0
		self.closest_intersection = ClosestCache()
		self.least_interconnected = ClosestCache()
		self.visit_count_on_last_growth = 0

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

	def reproduce(self, life: 'Life'):
		child = life.add_hotspot(self.pos + unit(Point.random()) * 100 * life.noise)

		child.visit_count_on_last_growth = self.visit_count_on_last_growth

		self.mass /= 2
		child.mass = self.mass

		child.vel = unit(Point.random()) * mag(self.vel) * random.random()
		self.vel -= child.vel


class HotspotSpawnNegotiator(DemandNegotiator):
	def __init__(self, life: 'Life') -> None:
		super().__init__('Hotspot Demand', Hotspot.cost)
		self.life = life

	def setup(self, scenario):
		return None

	def compute_demand(self, scenario, setup):
		speeds = [float(mag(hotspot.vel)) for hotspot in self.life.hotspots]
		if len(self.life.hotspots) > 0:
			min_speed, max_speed = min(speeds), max(speeds)
			return (min_speed / max_speed) * (.5 / (len(self.life.hotspots)))
		else:
			return 1

	def fund(self, scenario, fund: float, setup):
		self.life.add_hotspot(Point(unit(Point.random()) * 10000 * self.life.noise))
		return Hotspot.cost


class HotspotNegotiator(DemandNegotiator):
	def __init__(self, life: 'Life') -> None:
		super().__init__('Hotspot Growth')
		self.life = life

	def setup(self, scenario):
		return self.life.most_visited_hotspot(scenario)

	def compute_demand(self, scenario, mvh: Hotspot):
		if len(self.life.hotspots) > 0:
			current_count = mvh.closest_intersection.obj.visits
			prev_count = mvh.visit_count_on_last_growth
			if prev_count == 0:
				return 0 if current_count < 5 else 1
			relative_factor = (current_count / prev_count)
			return min(1, (relative_factor - 1) / (Hotspot.mass_multiplier - 1))
		return 0

	def fund(self, scenario, fund: float, mvh: Hotspot):
		mvh.mass *= Hotspot.mass_multiplier
		mvh.visit_count_on_last_growth = mvh.closest_intersection.obj.visits
		mvh.reproduce(self.life)
		return mvh.mass


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


class Life(RecursiveDemandJudge):
	def __init__(self, city: 'City'):
		super().__init__('Life Demand')
		self.friction = 0.0
		self.drag = 0.001
		self.noise = 0.01
		self.growth = 0
		self.hotspots = []  # type: List[Hotspot]
		self.momentum = np.ones([len(self.hotspots), len(self.hotspots)], np.float16)
		self.city = city

		self.add_negotiator(HotspotSpawnNegotiator(self)) \
			.add_negotiator(HotspotNegotiator(self)) \
			.add_negotiator(InterHotspotConnectivityNegotiator(self))

	def add_hotspot(self, spot: Point = None):
		if spot is None:
			spot = Point(self.city.infrastructure.most_used_road().a)
		hotspot = Hotspot(spot, Point(0, 0))
		self.hotspots.append(hotspot)
		self.cache_closest_intersection()
		self.dissipate([hotspot])
		return hotspot

	def dissipate(self, hotspots: List[Hotspot] = None):
		hotspots = hotspots or self.hotspots
		for a_i, a in enumerate(hotspots):
			a.force *= 0
			for b_i, b in enumerate(self.hotspots):
				if a_i != b_i:
					diff = a.pos - b.pos
					mag_diff = mag(diff)
					if mag_diff > 10:
						a.force += a.mass * b.mass * diff / mag_diff ** 3
			a.vel += (a.force + Point.random() * self.noise) / a.mass
			a_speed = mag(a.vel)
			a.vel *= max(0, (1 - self.friction - self.drag * a_speed)) * a.mass / (a.mass + self.growth)
			# a.mass += self.growth
			a.pos += a.vel / a.mass

	def reproduce(self):
		for parent in self.hotspots:
			if parent.mass > Hotspot.mature_mass:
				parent.reproduce(self)

	def tick(self):
		# self.reproduce()
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

	def most_visited_hotspot(self, tick: int = None):
		if len(self.hotspots) > 0:
			self.cache_closest_intersection(list(self.city.infrastructure.intersections), tick)
			hotspot_list = list(self.hotspots)
			visits = [hotspot.closest_intersection.obj.visits for hotspot in hotspot_list]
			return hotspot_list[int(np.argmax(visits))]
