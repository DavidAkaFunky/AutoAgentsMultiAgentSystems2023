# AutoAgentsMultiAgentSystems2023

Final grade: **18**

#

Before running any simulation we need to setup a configuration file containing the specifications of the simulation:

- `filename file`: If specified the output will be written in `file`
- `g x y`: `x` and `y` are the grid dimensions
- `f amount`: Describes que amount of food scattered in the grid
- `s num_steps`: `num_steps` Determines the number of steps for each episode
- `sit name`: It's possible to specifie different scenarios regarding the number of each agent type. `name` is the name associated to situation
- `p POLICY`: Policy used for agent priorities(better explanation in the project report). Associated to a situation
- `t tribe`: Defines a tribe. If present, the following agents belong to that tribe
- `a type greediness(n/y) starting_energy reproduction_threshold amount`: Defines an `amount` of agents with type `type`

Example of a configuration file at `configs/config_policies_part1.txt`:

```
filename test_policies_part1 
g 16 16
f 25
s 100
sit Random regular 50/50 greedy
p RANDOM
a regular n 25 40 12
a regular y 25 40 12
sit Benefit greedy 50/50 greedy
p BENEFIT_GREEDY
a regular n 25 40 12
a regular y 25 40 12
sit Benefit younger regular 50/50 greedy
p BENEFIT_YOUNGER
t t1
a regular n 25 40 12
t t2
a regular y 25 40 12
```

To run the simulation with a given configuration file, in the `/src` folder, execute :

`python .\simulate.py --episodes=<number_of_episodes> .\configs\config_policies_part1.txt`

Add `--render` option to visualize the environment representation.

