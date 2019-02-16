from typing import TypeVar, List

from multiagent_envs.swat.env import SWAT

ObType = TypeVar('ObType')


class Agent(object):
	def act(self, ob: ObType):
		raise NotImplementedError

	def reset(self):
		raise NotImplementedError


class DummyAgent(Agent):
	def act(self, ob: ObType):
		return 0

	def reset(self):
		return None


class MultiAgent(object):
	def __init__(self, agents: List[Agent]):
		self.agents = agents

	def act(self, obs):
		return [agent.act(ob) for agent, ob in zip(self.agents, obs)]

	def reset(self):
		for agent in self.agents:
			agent.reset()


if __name__ == '__main__':
	n_agents = 4
	swat = SWAT(n_agents)
	agents = MultiAgent([DummyAgent() for _ in range(n_agents)])
	episodes = 100

	swat.open()

	for episode in range(episodes):
		obs = swat.reset()
		agents.reset()

		while swat.is_open:
			actions = agents.act(obs)
			obs, rews, done, _ = swat.step(actions)
			if done:
				print('Done')
				break

	swat.close()
