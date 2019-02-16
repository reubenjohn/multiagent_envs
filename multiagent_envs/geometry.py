from typing import TYPE_CHECKING
from typing import Union

import numpy as np

if TYPE_CHECKING:
	from multiagent_envs.city_drivers.transport.common import Intersection


class Point(np.ndarray):
	def __new__(cls, x_or_point, y=None):
		obj = super().__new__(cls, (2,), np.float32)
		if y is not None:
			x_or_point = np.array([x_or_point, y], np.float32, copy=True)
		np.copyto(obj, x_or_point)
		return obj

	def __hash__(self) -> int:
		return hash(tuple(self))

	def __eq__(self, other):
		return np.all(np.equal(self, other))

	def __ne__(self, other):
		return not self.__eq__(other)

	@classmethod
	def random(cls):
		return np.random.random_sample(2) - .5


class Node(Point):
	def __new__(cls, x_or_point, y=None):
		return super().__new__(cls, x_or_point, y)


class Edge:
	def __init__(self, a: Union[Node, 'Intersection'], b: Union[Node, 'Intersection']):
		self.a, self.b = a, b
