from typing import TypeVar, List

from multiagent_envs.swat.env import SWAT, AgentAction, Goal

ObType = TypeVar('ObType')


class Agent(object):
	def act(self, ob: ObType):
		raise NotImplementedError

	def reset(self):
		raise NotImplementedError


class DummyAgent(Agent):
	def act(self, ob: ObType):
		return AgentAction(1, 1)

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
	swat = SWAT(n_agents, Goal(Goal.Verb(Goal.Verb.Type.REACH, None),
							   Goal.Noun(Goal.Noun.Type.POINT, None)))
	swat.display_interval = 1
	agents = MultiAgent([DummyAgent() for _ in range(n_agents)])
	episodes = 100

	swat.open()

	for episode in range(episodes):
		obs = swat.reset()
		agents.reset()

		while swat.is_open:
			swat.handle_input()
			swat.display()

			done = False
			for swat.n_step_over in range(int(swat.display_interval)):
				actions = agents.act(obs)
				obs, rews, done, _ = swat.step(actions)
				if done:
					break
			if done:
				break

	swat.close()
