from typing import List

import numpy as np


class DemandNegotiator:
	__slots__ = 'minimum_fund', 'name'

	def __init__(self, name: str = None, minimum_fund: float = 0) -> None:
		super().__init__()
		self.minimum_fund = minimum_fund
		if name is None:
			name = type(self).__name__
		self.name = name

	def setup(self, scenario):
		raise NotImplementedError

	def compute_demand(self, scenario, setup):
		raise NotImplementedError

	def fund(self, scenario, fund: float, setup):
		raise NotImplementedError


class SetupCache:
	def __init__(self, scenario=None, cache=None):
		self.scenario = scenario
		self.cache = cache


class DemandJudge:
	def __init__(self):
		self.best_negotiator = None  # type: DemandNegotiator
		self.negotiators = []  # type: List[DemandNegotiator]
		self.best_demand = None  # type: DemandNegotiator
		self.demands = []
		self.best_index = None
		self.setup_cache = SetupCache(None, [])

	def add_negotiator(self, negotiator: DemandNegotiator) -> 'DemandJudge':
		self.negotiators.append(negotiator)
		self.setup_cache.cache.append(None)
		self.demands.append(None)
		self.setup_cache.scenario = None
		return self

	def preside_negotiation(self, scenario):
		if self.setup_cache.scenario != scenario:
			for index, negotiator in enumerate(self.negotiators):
				self.setup_cache.cache[index] = negotiator.setup(scenario)

			self.demands = [negotiator.compute_demand(scenario, setup) for setup, negotiator in
							zip(self.setup_cache.cache, self.negotiators)]
			self.best_index = int(np.argmax(self.demands))
			self.best_demand = self.demands[self.best_index]
			self.best_negotiator = self.negotiators[self.best_index]

	def fund_best_negotiator(self, scenario, fund: float):
		self.preside_negotiation(scenario)
		if fund > self.best_negotiator.minimum_fund:
			return self.best_negotiator.fund(scenario, fund, self.setup_cache.cache[self.best_index])
		return 0
