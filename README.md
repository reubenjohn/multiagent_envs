Multiagent Environments
===
Collection of environments that act as a test bed for the development of Multi-Agent reinforcement learning

Environments
---
The environments have been catalogued below.  
Currently only one environment has been built so far.

### City Drivers
Each agent is spawned in a growing city and assigned goals.  
The city is procedurally generated in a timelapse like setting where roads are built instantaneously.  
All aspects of the city from city hotspots, goal assignment, road constructions, funding, road renovations and upscaling operates based on heuristics modeled after those of real cities.

**Key Features** implemented so far
1. Hotspots form as the city grows.
2. Roads are built to improve hotspot connectivity.
3. Vehicles navigate towards randomly assigned goals.
4. Out of the box classical AI implementation for vehicles to respect road safety.

**To Do**
1. Have vehicle goal assignment be biased towards more popular hotspots.
2. Have hotspots require funding to grow. With fund demand as a function of how many vehicles visit it.

`python3 -m multiagent_envs.city_drivers.main`

![City Drivers](multiagent_envs/city_drivers/screenshots/v0.1.0.png "City Drivers v0.1.0")

### SWAT (Work in progress)
Agents are assigned objectives in a procedurally generated shooting arena.  
This environment can be instantiated with a wide variety of modes(Free for all, Team death match, etc), scenarios(Initial player positions, weapon layouts, communications channels, etc), maps, etc.

**Key Features** implemented so far
1. All players are given a target location to which they have to move
2. Evolutionary algorithm baseline

`python3 -m multiagent_envs.swat.main`

![City Drivers](multiagent_envs/swat/screenshots/v0.1.0.png "SWAT v0.1.0")

### RoundAbout (Work in progress)
N cars in a round about must control their acceleration so as to maximise their collective speed.
This can be done through coordinated driving so as to prevent pile-ups.

`python3 -m multiagent_envs.swat.main`

![City Drivers](multiagent_envs/round_about/screenshots/v0.1.0.png "RoundAbout v0.1.0")