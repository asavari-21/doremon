# PPO + Domain Randomization for Portfolio Optimization

A PPO agent allocates across 10 NSE stocks + cash. It is trained in one of
several ways, then every strategy is tested on the same held-out real data.

**The question this answers (and what it does not):** *Does training on
randomized simulated markets (domain randomization, DR) make a portfolio
policy more robust on real data it has never seen, compared with training on
real history alone?* It does **not** claim to predict prices. DR can make a
policy robust; it cannot create a signal that isn't in the data.
