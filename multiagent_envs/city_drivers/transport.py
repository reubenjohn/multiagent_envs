import random
from typing import TYPE_CHECKING, Set, Dict, List

from multiagent_envs.const import X, Z
from multiagent_envs.geometry import Point, Node, Edge
from multiagent_envs.negotiator import DemandNegotiator, DemandJudge
from multiagent_envs.util import mag, cos

if TYPE_CHECKING:
	from multiagent_envs.city_drivers.city import City

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
	def breadth(self):
		return len(self.edges)

	@property
	def crookedness(self):
		cos_a_b_x = cos(self.a - self.b, X)
		return cos_a_b_x * np.sqrt(1 - cos_a_b_x ** 2)

	def next_lane_count(self):
		return self.breadth + (1 if self.breadth == 1 else 2)

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
		self.max_vel = Road.MAX_LANE_VEL * self.breadth


class Waypoint:
	def __init__(self, next_step: Intersection, remaining_dist: float):
		self.next_step = next_step
		self.dist = remaining_dist


class Router(dict):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def add_intersection(self, intersection: Intersection, adjs: Dict[Intersection, float]):
		intersection_map = {intersection: Waypoint(intersection, 0)}  # type: Dict[Intersection, Waypoint]
		for target in self:
			best_spt = Waypoint(next(adjs.__iter__()), 100000)
			for (adj, dist) in adjs.items():
				candidate_dist = dist + self[adj][target].dist
				if candidate_dist < best_spt.dist:
					best_spt.dist = candidate_dist
					best_spt.next_step = adj
			intersection_map[target] = best_spt
			if target in adjs:
				self[target][intersection] = Waypoint(intersection, best_spt.dist)
			else:
				self[target][intersection] = self[target][best_spt.next_step]
		self[intersection] = intersection_map


class Transport(DemandNegotiator):
	def __init__(self, city: 'City'):
		super().__init__('Vehicle Demand', Vehicle.cost)
		self.city = city
		self.vehicles = set()  # type: Set[Vehicle]

	def add_vehicle(self, road: Road):
		new_vehicle = Vehicle(self.city.infrastructure, road.a, road)
		self.assign_random_goal(new_vehicle)
		self.vehicles.add(new_vehicle)
		return new_vehicle

	def setup(self, scenario):
		return self.average_vehicle_speed() / Vehicle.max_speed

	def compute_demand(self, scenario, speed_leverage):
		return 1 if speed_leverage == 0 else speed_leverage

	def fund(self, scenario, fund: float, setup):
		if fund > Vehicle.cost:
			for attempt in range(1, 10):
				road = self.city.infrastructure.most_used_road()
				if len(road.vehicles) == 0 or road.length / len(road.vehicles) > Vehicle.min_safe_dist:
					self.add_vehicle(self.city.infrastructure.most_used_road())
					return Vehicle.cost
		return 0

	def tick(self):
		for vehicle in self.vehicles:
			if vehicle.tick():
				self.assign_random_goal(vehicle)

	def assign_random_goal(self, vehicle: 'Vehicle'):
		vehicle.set_global_dst(self.city.infrastructure.sample_intersection(exclude=vehicle.local_src))

	def average_vehicle_speed(self):
		if len(self.vehicles) > 0:
			return sum(vehicle.vel for vehicle in self.vehicles) / len(self.vehicles)
		return 0

	def average_vehicle_density_proportion(self):
		if len(self.vehicles) > 0:
			vehicle_roads = (vehicle.road for vehicle in self.vehicles)
			road_density = (len(road.vehicles) / (road.length * road.breadth) for road in vehicle_roads)
			density_to_max_density = sum(road_density) / (Vehicle.min_safe_dist * len(self.vehicles))
			return density_to_max_density
		return 0


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

	def __str__(self):
		return str((self.a, self.b))


class RoadExtensionNegotiator(DemandNegotiator):
	def __init__(self, city: 'City') -> None:
		super().__init__('Extension Demand', 10)
		self.city = city

	def setup(self, scenario):
		self.city.life.update_closest_intersection_cache(list(self.city.infrastructure.intersections), scenario)

	def compute_demand(self, scenario, setup):
		sum_hotspot_deprivation = sum(hotspot.closest.distance for hotspot in self.city.life.hotspots)
		sum_hotspot_coverage = sum(mag(hotspot.pos) for hotspot in self.city.life.hotspots) + .1
		return min(sum_hotspot_deprivation / sum_hotspot_coverage, 1)

	def fund(self, scenario, fund: float, setup):
		mdh = self.city.life.most_deprived_hotspot(list(self.city.infrastructure.intersections), scenario)
		if mdh is not None:
			diff = (mdh.pos - mdh.closest.intersection)
			diff_mag = mag(diff)
			length_to_build = min(fund, diff_mag)
			if fund > length_to_build:
				self.city.infrastructure.extend_road(mdh.closest.intersection,
													 mdh.closest.intersection + diff * length_to_build / diff_mag,
													 1, True)
				return length_to_build
		return 0


class RoadExpansionNegotiator(DemandNegotiator):
	def __init__(self, city: 'City') -> None:
		super().__init__('Expansion Demand')
		self.city = city

	def setup(self, scenario):
		return self.city.infrastructure.most_used_road()

	def compute_demand(self, scenario, mur: Road):
		vehicles_density = len(mur.vehicles) / (mur.length * mur.breadth)
		density_to_max_density = min(vehicles_density * Vehicle.min_safe_dist, 1)
		return density_to_max_density

	def fund(self, scenario, fund: float, mur):
		expansion_cost = mur.expansion_cost()
		if fund > expansion_cost:
			mur.expand()
			self.minimum_fund = mur.expansion_cost()
			return expansion_cost
		return 0


class Infrastructure(DemandNegotiator, DemandJudge):
	def __init__(self, city: 'City'):
		super().__init__()
		self.city = city
		root = Intersection(0, 0)
		self.nodes = set()  # type: Set[Node]
		self.edges = dict()  # type: Dict[Node, List[Node]]
		self.intersections = {root}  # type: Set[Intersection]
		self.roads = {}  # type: Dict[Joint, Road]

		self.router = Router()
		self.router.add_intersection(root, {})

		self.add_negotiator(RoadExtensionNegotiator(self.city))
		self.add_negotiator(RoadExpansionNegotiator(self.city))

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

	def setup(self, scenario):
		pass

	def compute_demand(self, scenario, setup):
		self.preside_negotiation(scenario)
		return self.best_demand

	def fund(self, scenario, fund: float, setup):
		return self.fund_best_negotiator(scenario, fund)

	def sample_intersection(self, exclude: Intersection = None):
		return random.sample(self.intersections.difference({exclude}), 1)[0]

	def sample_road(self):
		return random.sample(list(self.roads.values()), 1)[0]


class Vehicle:
	max_speed = 2
	cost = 40
	min_safe_dist = 4
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
		return self.infra.router[self.local_dst][self.global_dst].next_step

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
