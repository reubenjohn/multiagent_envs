import random
from typing import TYPE_CHECKING, Set, Dict, List

from multiagent_envs.const import X, Z
from multiagent_envs.geometry import Point, Node, Edge
from multiagent_envs.util import mag, cos

if TYPE_CHECKING:
	from multiagent_envs.city_drivers.city import City
	from multiagent_envs.city_drivers.life import Hotspot

import numpy as np


class Intersection(Point):
	pass


class Road(Edge):
	MAX_LANE_VEL = 0.5

	def __init__(self, network: 'Infrastructure', a: Intersection, b: Intersection, n_lanes: int = 1,
				 two_way: bool = False):
		super().__init__(a, b)
		self.max_vel = Road.MAX_LANE_VEL
		self.edges = self.two_way = None
		self.network = network
		self.usage = 0
		self.setup_edges(a, b, n_lanes, two_way)
		self.vehicles = set()  # type: Set[Vehicle]

	@property
	def length(self):
		return mag(self.a - self.b)

	@property
	def crookedness(self):
		cos_a_b_x = cos(self.a - self.b, X)
		return cos_a_b_x * np.sqrt(1 - cos_a_b_x ** 2)

	def next_lane_count(self):
		return len(self.edges) + (1 if len(self.edges) == 1 else 2)

	def expansion_cost(self):
		return self.length * self.next_lane_count() ** 2 * (1 + self.crookedness)

	def expand(self):
		self.setup_edges(self.a, self.b, self.next_lane_count())

	def setup_edges(self, a: Intersection, b: Intersection, n_lanes: int = 1, two_way: bool = False):
		assert n_lanes > 0
		if n_lanes != 1:
			assert two_way == False and n_lanes % 2 == 0
		self.two_way = two_way
		forward = b - a
		right = np.cross(np.array([forward[0], forward[1], 0]), Z)[:2]
		right /= np.linalg.norm(right)
		self.edges = []
		if n_lanes == 1:
			src, dst = self.network.to_node(a), self.network.to_node(b)
			self.edges = [Edge(src, dst)]
			if two_way:
				self.edges.append(Edge(dst, src))
		else:
			for middle_offset in np.arange(-(n_lanes - 1) / 2, (n_lanes - 1) / 2 + 1, 1):
				if middle_offset < 0:
					a, b = self.a, self.b
				else:
					a, b = self.b, self.a
				self.edges.append(
					Edge(self.network.to_node(a + right * middle_offset),
						 self.network.to_node(b + right * middle_offset)))
		self.max_vel = Road.MAX_LANE_VEL * len(self.edges)


class Waypoint:
	def __init__(self, next_step: Intersection, remaining_dist: float):
		self.next_step = next_step
		self.dist = remaining_dist


class Router(object):
	def __init__(self):
		self.spts = dict()  # type: Dict[Intersection, Dict[Intersection, Waypoint]]

	def add_intersection(self, intersection: Intersection, adjs: Dict[Intersection, float]):
		intersection_map = {intersection: Waypoint(intersection, 0)}  # type: Dict[Intersection, Waypoint]
		for target in self.spts:
			best_spt = Waypoint(next(adjs.__iter__()), 100000)
			for (adj, dist) in adjs.items():
				candidate_dist = dist + self.spts[adj][target].dist
				if candidate_dist < best_spt.dist:
					best_spt.dist = candidate_dist
					best_spt.next_step = adj
			intersection_map[target] = best_spt
			if target in adjs:
				self.spts[target][intersection] = Waypoint(intersection, best_spt.dist)
			else:
				self.spts[target][intersection] = self.spts[target][best_spt.next_step]
		self.spts[intersection] = intersection_map


class Transport:
	def __init__(self, city: 'City'):
		self.city = city
		self.vehicles = set()  # type: Set[Vehicle]

	def add_vehicle(self, road: Road):
		if not any(vehicle.road == road for vehicle in self.vehicles):
			new_vehicle = Vehicle(self.city.infrastructure, road.a, road)
			self.assign_random_goal(new_vehicle)
			self.vehicles.add(new_vehicle)
			return new_vehicle
		return None

	def fund(self, fund):
		if fund > Vehicle.cost:
			if self.add_vehicle(self.city.infrastructure.most_used_road()) is not None:
				return Vehicle.cost
			else:
				return 0
		return 0

	def tick(self):
		for vehicle in self.vehicles:
			if vehicle.tick():
				self.assign_random_goal(vehicle)

	def assign_random_goal(self, vehicle: 'Vehicle'):
		vehicle.set_global_dst(self.city.infrastructure.sample_intersection(exclude=vehicle.local_src))


class Joint:
	def __init__(self, a: Intersection, b: Intersection):
		self.a = a
		self.b = b

	def __hash__(self) -> int:
		return hash(self.a) + hash(self.b)

	def __eq__(self, o: object) -> bool:
		if isinstance(o, Joint):
			return self.a == o.a and self.b == o.b or self.a == o.b and self.b == o.a
		else:
			return False


class Infrastructure:
	def __init__(self, city: 'City'):
		self.city = city
		root = Intersection(0, 0)
		self.nodes = set()  # type: Set[Node]
		self.edges = dict()  # type: Dict[Node, List[Node]]
		self.intersections = {root}  # type: Set[Intersection]
		self.roads = {}  # type: Dict[Joint, Road]

		self.router = Router()
		self.router.add_intersection(root, {})

		self.candidates = None
		self.mdh = None

		self.extend_road(root, Intersection(0, 10), 1, True)

	def extend_road(self, src: Intersection, dst: Intersection, n_lanes, two_way: bool = True):
		assert src in self.intersections and dst not in self.intersections

		self.intersections.add(dst)
		self.router.add_intersection(dst, {src: mag(src - dst)})

		if src not in self.roads:
			self.roads[Joint(src, dst)] = dict()  # type: Dict[Intersection, Road]
		self.roads[Joint(src, dst)] = Road(self, src, dst, n_lanes, two_way)

	def to_node(self, intersection: Intersection):
		return self.node(Node(intersection))

	def node(self, node: Node):
		if not node in self.nodes:
			self.nodes.add(node)
		return node

	def most_used_road(self):
		roads = [road for road in self.roads.values()]
		road_usages = np.array([road.usage for road in roads])
		max_arg = road_usages.argmax()
		return roads[max_arg]

	def road_cost_benefit_ratio(self, road):
		return float(road.expansion_cost() / (500 * (road.usage + 0.1) / self.city.ticks))

	def intersection_cost_benefit_ratio(self, hotspot: 'Hotspot'):
		diff = (hotspot.pos - hotspot.closest.intersection)
		diff_s = mag(diff)
		if diff_s < 10:
			return np.inf
		return 20 / diff_s

	def fund(self, fund: float):
		mur = self.most_used_road()
		mdh = self.city.life.most_deprived_hotspot(self.intersections)

		if mdh is not None and self.intersection_cost_benefit_ratio(mdh) < self.road_cost_benefit_ratio(mur):
			self.mdh = mdh
			if fund > 10:
				diff = (mdh.pos - mdh.closest.intersection)
				diff_mag = mag(diff)
				length_to_build = min(fund, diff_mag)
				self.extend_road(Intersection(mdh.closest.intersection),
								 mdh.closest.intersection + diff * length_to_build / diff_mag, 1, True)
				return length_to_build
		else:
			expansion_cost = mur.expansion_cost()
			if fund > expansion_cost:
				mur.expand()
				return expansion_cost
		return 0

	def sample_intersection(self, exclude: Intersection = None):
		return random.sample(self.intersections.difference({exclude}), 1)[0]


class Vehicle:
	cost = 1
	min_safe_dist = 2
	CHILLING = 0
	RACING = 5
	FREAKING = 8
	BASKING_IN_GLORY = 10

	def __init__(self, infra: Infrastructure, intersection: Intersection, road: Road, pos: float = None):
		self.infra = infra
		self.global_dst = None
		self.local_src = self.local_dst = intersection
		self.road = road  # type: Road
		self.road.vehicles.add(self)
		self.vel = 0.0  # type: float
		self.pos = road.length / 2 if pos is None else pos  # type: float
		self.state = Vehicle.CHILLING

	def set_global_dst(self, global_dst: Intersection):
		self.global_dst = global_dst
		self.target_next_waypoint()

	def tick(self):
		self.road.usage += 1
		if self.global_dst is not None:
			target = self.control()
			self.vel += max(min(target - self.vel, 0.1), -0.1)
			self.pos += self.vel
			if self.pos > self.road.length:
				self.pos -= self.road.length
				if self.local_dst == self.global_dst:
					self.local_src = self.local_dst
					self.global_dst = None
					self.vel = 0
					self.pos = 0
					self.state = Vehicle.BASKING_IN_GLORY
					return True
				else:
					self.target_next_waypoint()
		return False

	def compute_local_dst(self):
		return self.infra.router.spts[self.local_dst][self.global_dst].next_step

	def target_next_waypoint(self):
		self.local_src, self.local_dst = self.local_dst, self.compute_local_dst()
		self.road.vehicles.remove(self)
		self.road = self.infra.roads[Joint(self.local_src, self.local_dst)]
		self.road.vehicles.add(self)  # Help other vehicles find me

	def control(self):
		self.state = Vehicle.RACING
		stop_ratio = self.pos / self.road.length
		upcoming_vehicle_positions = [
			other.pos for other in self.road.vehicles
			if other != self and other.local_src == self.local_src and other.pos > self.pos
		]
		if len(upcoming_vehicle_positions) > 0:
			upcoming_vehicle_stop_ratio = self.min_safe_dist / (min(upcoming_vehicle_positions) - self.pos)
			if upcoming_vehicle_stop_ratio - stop_ratio > .2:
				self.state = Vehicle.FREAKING
			stop_ratio = max(stop_ratio, upcoming_vehicle_stop_ratio)
		return max(self.road.max_vel / 10,
				   self.road.max_vel * (1 - stop_ratio ** 2))
