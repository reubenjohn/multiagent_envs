from typing import TYPE_CHECKING, Set

from multiagent_envs.city_drivers.transport.common import Joint, Intersection
from multiagent_envs.negotiator import DemandNegotiator

if TYPE_CHECKING:
	from multiagent_envs.city_drivers.transport.infrastructure import Road, Infrastructure
	from multiagent_envs.city_drivers.city import City


class Transport(DemandNegotiator):
	def __init__(self, city: 'City'):
		super().__init__('Vehicle Demand', Vehicle.cost)
		self.city = city
		self.vehicles = set()  # type: Set[Vehicle]

	def add_vehicle(self, road: 'Road'):
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


class Vehicle:
	max_speed = 2
	cost = 40
	min_safe_dist = 4
	CHILLING = 0
	RACING = 5
	FREAKING = 8
	BASKING_IN_GLORY = 10

	def __init__(self, infra: 'Infrastructure', intersection: Intersection, road: 'Road', pos: float = None):
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
