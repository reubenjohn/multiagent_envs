from typing import Dict

from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from multiagent_envs.city_drivers.transport import Intersection


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
