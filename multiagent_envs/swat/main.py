from typing import TypeVar, List, Generic

import numpy as np

from multiagent_envs.swat.env import SWAT, AgentAction, Goal, AgentState, AgentDebug

ObType = TypeVar('ObType')

sample = np.random.random_sample


class Agent(object):
	def __init__(self) -> None:
		super().__init__()
		self.debug = None

	def act(self, ob: ObType):
		raise NotImplementedError

	def reset(self, debug: AgentDebug):
		self.debug = debug


class EvolutionaryAgent(Agent):
	def __init__(self, weights=None):
		super().__init__()
		self.weights = sample([5, 4]) if weights is None else weights
		self.ob = None

	def act(self, ob: AgentState):
		self.ob = ob
		ob_vector = np.concatenate([ob.pos, [ob.direction], ob.goal.noun.adjective])
		logits = np.matmul(np.expand_dims(ob_vector, 0), self.weights)[0]
		direction = AgentState.normalize_direction(int(np.argmax(logits)))
		return AgentAction(direction, 0)

	def get_mutated_weights(self):
		return self.weights + sample([5, 4]) * .1, tuple((np.array(self.debug.color) + sample(3) * .1).tolist())

	def reset(self, debug: AgentDebug):
		super().reset(debug)
		self.debug.color = tuple((sample(3) * .24 + .75).tolist())


AgentType = TypeVar('AgentType', bound=Agent)


class MultiAgent(Generic[AgentType]):
	def __init__(self, agents: List[AgentType]):
		self.agents = agents

	def act(self, observations):
		return [agent.act(ob) for agent, ob in zip(self.agents, observations)]

	def reset(self, agent_debugs: List[AgentDebug]):
		for agent, debug in zip(self.agents, agent_debugs):
			agent.reset(debug)


class MultiEvolutionaryAgent(MultiAgent):
	def __init__(self, agents: List[EvolutionaryAgent], selection_rate: float = .1, mutation_rate: float = .1):
		super().__init__(agents)
		self.selection_rate, self.mutation_rate = selection_rate, mutation_rate

	def reset(self, agent_debug: List[AgentDebug]):
		super().reset(agent_debug)
		if self.agents[0].ob is not None:
			fitness_order = np.argsort([-agent.ob.health for agent in self.agents])

			fitness_boundary = int(len(self.agents) * self.selection_rate)
			mutated_parent_traits = [agent.get_mutated_weights() for agent in
									 (self.agents[agent_i] for agent_i in
									  fitness_order[:fitness_boundary])]
			for agent_i in range(fitness_boundary, len(self.agents)):
				inheriting_agent = self.agents[fitness_order[agent_i]]
				weights, color = mutated_parent_traits[(agent_i - fitness_boundary) % fitness_boundary]
				inheriting_agent.weights = weights
				inheriting_agent.debug.color = color


if __name__ == '__main__':
	n_agents = 50
	swat = SWAT(n_agents, Goal(Goal.Verb(Goal.Verb.Type.REACH, None),
							   Goal.Noun(Goal.Noun.Type.POINT, None)), (25, 25))
	swat.display_interval = 1
	multi_agent = MultiEvolutionaryAgent([EvolutionaryAgent() for _ in range(n_agents)])
	episodes = 1000

	swat.open()

	for episode in range(episodes):
		obs, debugs = swat.reset()
		multi_agent.reset(debugs)

		while swat.is_open:
			swat.display()
			swat.handle_input()

			done = False
			for swat.n_step_over in range(int(swat.display_interval)):
				actions = multi_agent.act(obs)
				obs, rews, done, _ = swat.step(actions)
				if done or not swat.is_open:
					break
			if done:
				break
		if not swat.is_open:
			break

	swat.close()
