from typing import Dict, Set
from typing import TYPE_CHECKING

from multiagent_envs.util import mag

if TYPE_CHECKING:
	from multiagent_envs.city_drivers.transport.common import Intersection


class Waypoint:
	def __init__(self, next_step: 'Intersection', remaining_dist: float):
		self.next_step = next_step
		self.dist = remaining_dist


class Router(dict):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def add_intersection(self, intersection: 'Intersection', adjs: Dict['Intersection', float]):
		intersection_map = {intersection: Waypoint(intersection, 0)}  # type: Dict['Intersection', Waypoint]
		for target in self:
			best_spt = Waypoint(next(adjs.__iter__()), 100000)
			for adj, dist in adjs.items():
				candidate_dist = dist + self[adj][target].dist
				if candidate_dist < best_spt.dist:
					best_spt.dist = candidate_dist
					best_spt.next_step = adj
			intersection_map[target] = best_spt
			if target in adjs:
				self[target][intersection] = Waypoint(intersection, best_spt.dist)
			else:
				self[target][intersection] = Waypoint(self[target][best_spt.next_step].next_step, best_spt.dist)
		self[intersection] = intersection_map

	def connect_intersections(self, a: 'Intersection', b: 'Intersection', intersections: Set['Intersection']):
		a_b_separation = mag(a - b)
		self[a][b].dist = self[b][a].dist = a_b_separation
		self[a][b].next_step = b
		self[b][a].next_step = a

		# Partition all other intersections based on distances to a and b
		a_set = set()
		b_set = set()
		for inter in intersections:
			if self[inter][a].dist < self[inter][b].dist:
				a_set.add(inter)
			else:
				b_set.add(inter)

		for a_inter in a_set - {b}:
			for b_inter in b_set:
				newly_offered_separation = self[a_inter][a].dist + a_b_separation + self[b][b_inter].dist
				if newly_offered_separation < self[a_inter][b_inter].dist:
					self[a_inter][b_inter].dist = self[b_inter][a_inter].dist = newly_offered_separation
					self[a_inter][b_inter].next_step = self[a_inter][a].next_step if a_inter != a else b
					self[b_inter][a_inter].next_step = self[b_inter][b].next_step if b_inter != b else a

	def __getitem__(self, k) -> Dict['Intersection', Waypoint]:
		return super().__getitem__(k)
