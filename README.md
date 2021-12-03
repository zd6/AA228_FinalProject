# AA228_FinalProject
## Single Agent Pick-Ups of Stochastic Packages with Traffic Impediment Using Reinforcement Learning and Value Iteration
### Zhengguan (Gary) Dai and David Wu
In the shipping industry, picking up outgoing packages from customers in the most time and cost efficient way is important for business success. This paper solves a simplified version of cargo pickup by modeling the problem as a Markov Decision Process in a grid world with one agent where the full state is observable. Packages pop up in the grid with spatial probabilities that mimic reality; fewer packages appear in the rural areas. The agent incurs greater cost after business hours but is able to travel faster due to there being less traffic that impedes the transition from the current location to the next. We assume that all packages give the same reward. We use the following 5 methods: value iteration, Sample-based Value Iteration (SVI), Deep Q-learning using Neural networks (DQN) to approximate the action value function, random, and greedy. Results show that SVI achieves the highest average daily reward, albeit requiring the longest runtime. DQN is sometimes capable of achieving results comparable to SVI, and only needs half the runtime of SVI, but is less consistent. Overall, we find that for problems with smaller state spaces, an exact solution method would be optimal; whereas for problems with larger state spaces, a model-free method would scale better and thus be better suited for complex real-world applications. 
