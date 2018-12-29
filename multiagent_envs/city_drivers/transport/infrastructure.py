import random
from typing import TYPE_CHECKING, Set, Dict, List

from multiagent_envs.city_drivers.transport.common import Joint, Intersection

if TYPE_CHECKING:
	from multiagent_envs.city_drivers.city import City

import numpy as np

from multiagent_envs.city_drivers.transport.driver import Router
from multiagent_envs.city_drivers.transport.transport import Vehicle
from multiagent_envs.const import X, Z
from multiagent_envs.geometry import Edge, Node
from multiagent_envs.negotiator import DemandNegotiator, RecursiveDemandJudge
from multiagent_envs.util import mag, cos


class Road(Edge):
	min_length = 10
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

		assert self.length != 0

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


class RoadExtensionNegotiator(DemandNegotiator):
	def __init__(self, city: 'City') -> None:
		super().__init__('Extension Demand', 10)
		self.city = city

	def setup(self, scenario):
		self.city.life.cache_closest_intersection(list(self.city.infrastructure.intersections), scenario)

	def compute_demand(self, scenario, setup):
		sum_hotspot_deprivation = sum(hotspot.closest_intersection.separation for hotspot in self.city.life.hotspots)
		sum_hotspot_coverage = sum(mag(hotspot.pos) for hotspot in self.city.life.hotspots) + .1
		return min(sum_hotspot_deprivation / sum_hotspot_coverage, 1)

	def fund(self, scenario, fund: float, setup):
		mdh = self.city.life.most_deprived_hotspot(list(self.city.infrastructure.intersections), scenario)
		if mdh is not None:
			diff = (mdh.pos - mdh.closest_intersection.obj)
			diff_mag = mag(diff)
			length_to_build = min(fund, diff_mag)
			self.city.infrastructure.extend_road(mdh.closest_intersection.obj,
												 mdh.closest_intersection.obj + diff * length_to_build / diff_mag,
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

	def fund(self, scenario, fund: float, mur: Road):
		expansion_cost = mur.expansion_cost()
		if fund > expansion_cost:
			mur.expand()
			self.minimum_fund = mur.expansion_cost()
			return expansion_cost
		return 0


class Infrastructure(RecursiveDemandJudge):
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

		road = Road(self, src, dst, n_lanes, two_way)
		self.roads[Joint(src, dst)] = road

		self.router.add_intersection(dst, {src: mag(src - dst)})

	def connect_intersections(self, src: Intersection, dst: Intersection, n_lanes, two_way: bool = True):
		assert src in self.intersections and dst in self.intersections

		road = Road(self, src, dst, n_lanes, two_way)
		self.roads[Joint(src, dst)] = road

		self.router.connect_intersections(src, dst, self.intersections)

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

	def sample_intersection(self, exclude: Intersection = None):
		return random.sample(self.intersections.difference({exclude}), 1)[0]

	def sample_road(self):
		return random.sample(list(self.roads.values()), 1)[0]
