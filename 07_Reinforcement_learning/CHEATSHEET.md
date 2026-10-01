# 07 · Cheat Sheet: Reinforcement Learning

> **Notebooks:** [rl_basics](rl_basics.ipynb) · [dqn_and_actor_critic](dqn_and_actor_critic.ipynb)
>
> **Full explanations:** [README](README.md)

## Core definitions

- **MDP** $(S, A, P, R, \gamma)$. Markov property: the future depends only on the current state.
- **Return:** $G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$, where $\gamma \in [0, 1)$ discounts future reward. The effective horizon is about $\frac{1}{1-\gamma}$.
- **State value:** $V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s]$
- **Action value:** $Q^\pi(s, a) = \mathbb{E}_\pi[G_t \mid s_t = s, a_t = a]$
- **Advantage:** $A(s,a) = Q(s,a) - V(s)$

## Bellman equations

| | Equation |
|---|---|
| Expectation | $V^\pi(s) = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a)\,[R + \gamma V^\pi(s')]$ |
| Optimality | $V^*(s) = \max_a \sum_{s'} P(s' \mid s, a)\,[R + \gamma V^*(s')]$ |
| Q-optimality | $Q^*(s, a) = \mathbb{E}[R + \gamma \max_{a'} Q^*(s', a')]$ |

**Value iteration:** apply the optimality update to every state until $\max_s \lvert \Delta V \rvert < \epsilon$, then act greedily: $\pi^*(s) = \arg\max_a [\dots]$. It needs a known model $P$.

## Algorithms

| | Learns | Update | On/off-policy | Actions |
|---|---|---|---|---|
| **REINFORCE** | $\pi_\theta$ | $\nabla J = \mathbb{E}[\sum_t \nabla \log \pi_\theta(a_t \mid s_t)\,G_t]$ | On | Discrete or continuous |
| **Q-learning** | $Q$ | $Q \leftarrow Q + \alpha[r + \gamma \max_{a'} Q(s', a') - Q]$ | Off | Discrete |
| **DQN** | $Q_\theta$ | $\text{Huber}\big(Q_\theta(s,a),\; r + \gamma \max_{a'} Q_{\theta^-}(s', a')\big)$ | Off | Discrete |
| **A2C** | $\pi_\theta$ and $V_w$ | Actor: $-\log\pi(a \mid s)\,A_t$; critic: $(G_t - V(s))^2$ | On | Discrete or continuous |

**What makes DQN stable:** an experience **replay buffer** (breaks correlation), a **target network** $\theta^-$ (stable targets, updated by hard copy or soft $\tau$), and ε-greedy exploration with decaying ε.

**Variance reduction for policy gradients:** subtract a baseline $b(s)$ (usually $V(s)$, which gives the advantage), normalise returns, use an entropy bonus to keep exploring, and use more samples per update.

## Hyperparameters

| Hyperparameter | Typical value |
|---|---|
| Discount $\gamma$ | 0.99 |
| Learning rate | 1e-3 to 1e-4 |
| ε (exploration) | 1.0 (or 0.9) decaying to 0.05 |
| Replay buffer size | 1e4 to 1e6 |
| Batch size | 32–256 |
| Target network update | $\tau$ = 0.005 (soft), or a hard copy every $N$ steps |

## Pitfalls

- Don't bootstrap from terminal states. Use $y = r$ when `terminated` is true (a time-limit `truncated` is not terminal).
- RL curves are **noisy**. Plot a moving average and run several seeds before concluding anything.
- Reward scale matters. Very large rewards destabilise value learning.
- Detach the advantage (or target) so the actor loss doesn't update the critic.
- Policy gradient methods are sample-inefficient. Expect thousands of episodes even on CartPole.
