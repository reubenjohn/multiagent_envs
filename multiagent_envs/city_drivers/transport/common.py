from multiagent_envs.geometry import Point


class Intersection(Point):
	pass


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
