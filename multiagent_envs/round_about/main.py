from typing import TypeVar, List, Generic

import numpy as np

from multiagent_envs.round_about.env import RoundAbout, AgentOb, AgentAction, AgentDebug
from multiagent_envs.util import unit

ObType = TypeVar('ObType')

sample = np.random.random_sample


class Agent(object):
	def __init__(self, debug: AgentDebug) -> None:
		super().__init__()
		self.debug = debug

	def act(self, ob: ObType):
		raise NotImplementedError

	def reset(self):
		pass


class EvolutionaryAgent(Agent):
	def __init__(self, max_acc: float, weights=None, n_inputs=5, n_outputs=3, debug=None):
		self.max_acc = max_acc
		self.n_inputs, self.n_outputs = n_inputs + 2, n_outputs
		self.weights = sample([self.n_inputs, n_outputs]) * 2 - 1 if weights is None else weights
		super().__init__(debug)
		self.ob = None
		self.debug.true_color = sample(3)
		self.debug.youth = 1
		self.debug.gene_novelty = 1

	def act(self, ob: AgentOb):
		self.ob = ob
		ob_vector = np.array([ob.vel, ob.behind_dist, ob.behind_vel, ob.forward_dist, ob.forward_vel, 1, np.random.random_sample()])
		output_logits = np.matmul(np.expand_dims(ob_vector, 0), self.weights)[0]
		choice = int(np.argmax(output_logits))
		acc = [-self.max_acc, 0, self.max_acc][choice]
		return AgentAction(acc)

	def mutation(self, mutation_rate, size):
		return np.random.normal(0, mutation_rate, size)

	def get_mutated_traits(self, mutation_rate: float):
		mutated_color = self.debug.true_color + self.mutation(mutation_rate, 3)
		true_color = unit(mutated_color)
		return np.clip(self.weights + self.mutation(mutation_rate, [self.n_inputs, self.n_outputs]), -1, 1), \
			   true_color, \
			   self.debug.gene_novelty * (1 - mutation_rate / 4), \
			   self.debug.youth * (1 - mutation_rate / 4)

	def update_derived_traits(self):
		self.debug.color = self.debug.true_color.tolist()
		self.debug.gene_novelty_color = (np.ones(3) * self.debug.gene_novelty).tolist()
		self.debug.youth_color = (np.ones(3) * self.debug.youth).tolist()

	def inherit_traits(self, traits):
		self.weights, self.debug.true_color, self.debug.gene_novelty, parent_youth = traits
		self.update_derived_traits()

	def preserve(self, traits):
		weights, self.debug.true_color, self.debug.gene_novelty, self.debug.youth = traits
		self.update_derived_traits()

	def reset(self):
		super().reset()
		self.debug.true_color = sample(3)
		self.debug.youth = 1
		self.debug.gene_novelty = 1
		self.update_derived_traits()


AgentType = TypeVar('AgentType', bound=Agent)


class MultiAgent(Generic[AgentType]):
	def __init__(self, agents: List[AgentType]):
		self.agents = agents

	def act(self, observations):
		return [agent.act(ob) for agent, ob in zip(self.agents, observations)]

	def reset(self, agent_debugs: List[AgentDebug]):
		for agent, debug in zip(self.agents, agent_debugs):
			agent.reset()


class MultiEvolutionaryAgent(MultiAgent):
	def __init__(self, agents: List[EvolutionaryAgent],
				 selection_rate: float = .8, mutation_rate: float = .01):
		super().__init__(agents)
		self.selection_rate, self.mutation_rate = selection_rate, mutation_rate
		self.values = np.zeros(len(self.agents))

	def reset(self, agent_debug: List[AgentDebug]):
		fitness_order = np.argsort(-self.values)
		fitness_boundary = int(len(self.agents) * self.selection_rate)
		mutated_parent_traits = [agent.get_mutated_traits(self.mutation_rate) for agent in
								 (self.agents[agent_i] for agent_i in fitness_order[:fitness_boundary])]

		super().reset(agent_debug)

		for agent_i in range(fitness_boundary):
			parent = self.agents[fitness_order[agent_i]]
			mutated_traits = mutated_parent_traits[agent_i]
			parent.preserve(mutated_traits)

		for agent_i in range(fitness_boundary, len(self.agents)):
			inheriting_agent = self.agents[fitness_order[agent_i]]
			mutated_traits = mutated_parent_traits[(agent_i - fitness_boundary) % fitness_boundary]
			inheriting_agent.inherit_traits(mutated_traits)

		self.values = np.zeros(len(self.agents))

	def accumulate_rewards(self, rewards: List[float]):
		np.add(self.values, np.array(rewards), out=self.values)


if __name__ == '__main__':
	n_agents = 5
	env = RoundAbout(n_agents)
	env.display_interval = 1
	multi_agent = MultiEvolutionaryAgent(
		[EvolutionaryAgent(env.max_acc, debug=agent_debug) for agent_debug in env.agent_debugs],
		selection_rate=.8, mutation_rate=.05
	)
	episodes = 10000

	env.open()

	for episode in range(episodes):
		multi_agent.reset(env.agent_debugs)
		obs = env.reset()

		while env.is_open:
			env.handle_input()

			done = False
			for env.n_step_over in range(int(env.display_interval)):
				actions = multi_agent.act(obs)
				obs, rews, done, _ = env.step(actions)
				multi_agent.accumulate_rewards(rews)
				if done or not env.is_open:
					break
			if env.is_open:
				env.display()
			if done:
				break
		if not env.is_open:
			break

	env.close()
