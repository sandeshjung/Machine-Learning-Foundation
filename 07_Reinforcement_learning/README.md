# Reinforcement Learning

## Fundamentals & Policy Gradients

### Introduction to Reinforcement Learning

#### The Agent-Environment Paradigm

Reinforcement Learning represents a computational approach to understanding and automating goal-directed learning and decision-making. Unlike supervised learning, where the correct action is provided, or unsupervised learning, where the goal is to find hidden structure, RL focuses on learning through interaction with an environment to maximize cumulative reward.

The fundamental interaction occurs through the **agent-environment loop**:

```
State(t) → Agent → Action(t) → Environment → Reward(t+1), State(t+1)

```

This cyclical process continues until either:

-   A terminal state is reached (episodic tasks)
-   A maximum number of steps is executed
-   The learning process is manually terminated

#### Key Components and Terminology

**Agent ($\large \mathcal{A}$)**: The learner and decision-maker that perceives the environment and selects actions.

**Environment ($\large \mathcal{E}$)**: The external system with which the agent interacts. It encompasses everything outside the agent's direct control.

**State ($\large s \in \mathcal{S}$)**: A complete description of the world's current configuration that contains all information necessary for decision-making. The state space $\large \mathcal{S}$ contains all possible states.

**Action ($\large a \in \mathcal{A}$)**: A choice made by the agent that affects the environment. The action space $\large \mathcal{A}$ may be discrete (finite set) or continuous.

**Reward ($\large r \in \mathcal{R} \subseteq \mathbb{R}$)**: A scalar signal that indicates the immediate desirability of the agent's action. Rewards provide the only feedback mechanism for learning.

**Policy ($\large \pi$)**: The agent's strategy for selecting actions. It defines the mapping from states to actions or action probabilities.

**Value Function**: A function that estimates the expected long-term reward from states or state-action pairs under a given policy.

#### The RL Problem Statement

The central problem in RL can be formally stated as:

> **Given an environment modeled as an MDP, find a policy $\large \pi^*$ that maximizes the expected cumulative discounted reward.**

Mathematically, this translates to:

$$\large 
\pi^* = \arg\max_{\pi} \mathbb{E}_{\tau \sim \pi}\left[\sum_{t=0}^{T} \gamma^t R_{t+1} \mid S_0\right]
$$

where:

-   $\large \tau$ represents a trajectory (sequence of states, actions, and rewards)
-   $\large T$ is the time horizon (finite for episodic tasks, infinite for continuing tasks)
-   $\large \gamma \in [0,1]$ is the discount factor


### Mathematical Foundations: Markov Decision Processes

#### Formal MDP Definition

A Markov Decision Process provides the mathematical framework for modeling sequential decision-making problems. An MDP is formally defined by the tuple:

$$\large 
\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)
$$

where:

**State Space ($\large \mathcal{S}$)**: The set of all possible states. This can be:

-   Finite: $\large \mathcal{S} = {s_1, s_2, \ldots, s_n}$
-   Countably infinite: $\large \mathcal{S} = {s_1, s_2, s_3, \ldots}$
-   Continuous: $\large \mathcal{S} \subseteq \mathbb{R}^n$

**Action Space ($\large \mathcal{A}$)**: The set of all possible actions. Can be state-dependent $\large \mathcal{A}(s)$ or global $\large \mathcal{A}$.

**Transition Dynamics ($\large \mathcal{P}$)**: The probability distribution over next states given current state and action: 

$$\large 
\mathcal{P}(s' \mid s, a) = \Pr(S_{t+1} = s' \mid S_t = s, A_t = a)
$$

**Reward Function ($\large \mathcal{R}$)**: The expected immediate reward: 

$$\large 
\mathcal{R}(s, a) = \mathbb{E}[R_{t+1} \mid S_t = s, A_t = a]
$$

**Discount Factor ($\large \gamma$)**: A value in $\large [0, 1]$ that determines the present value of future rewards.

#### The Markov Property

The Markov property is the cornerstone assumption that makes RL mathematically tractable:

$$\large 
\Pr(S_{t+1} = s', R_{t+1} = r \mid S_0, A_0, R_1, \ldots, S_t, A_t) = \Pr(S_{t+1} = s', R_{t+1} = r \mid S_t, A_t)
$$

This states that the future depends only on the present state and action, not on the entire history. This assumption allows us to make optimal decisions based solely on current information.

#### Return and Discount Factor

The **return** $\large G_t$ represents the cumulative discounted reward from time $\large t$ onwards:

$$\large 
G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
$$

The discount factor $\large \gamma$ serves multiple purposes:

1.  **Mathematical Convenience**: Ensures convergence of infinite sums when $\large \gamma < 1$
2.  **Temporal Preference**: Models preference for immediate rewards over delayed ones
3.  **Uncertainty Modeling**: Accounts for uncertainty about the future

**Special Cases**:

-   $\large \gamma = 0$: Myopic agent (only immediate rewards matter)
-   $\large \gamma = 1$: Far-sighted agent (all future rewards equally important)
-   $\large \gamma \to 1$: Approaches undiscounted case

### Policies: The Heart of Decision Making

#### Deterministic vs. Stochastic Policies

**Deterministic Policy**: $\large \pi: \mathcal{S} \to \mathcal{A}$ 

$$\large 
a = \pi(s)
$$

**Stochastic Policy**: $\large \pi: \mathcal{S} \times \mathcal{A} \to [0, 1]$ 

$$\large 
\pi(a \mid s) = \Pr(A_t = a \mid S_t = s)
$$

with the constraint: $\large \sum_{a \in \mathcal{A}} \pi(a \mid s) = 1$ for all $\large s \in \mathcal{S}$.

#### Policy Representation

**Tabular Representation**: For small, discrete state and action spaces 

$$\large 
\pi(a \mid s) = \begin{cases} 0.7 & \text{if } a = a_1 \ 0.3 & \text{if } a = a_2 \ 0 & \text{otherwise} \end{cases}
$$

**Parametric Representation**: For large or continuous spaces 

$$\large 
\pi_\theta(a \mid s) = \frac{\exp(\phi(s, a)^T \theta)}{\sum_{a'} \exp(\phi(s, a')^T \theta)}
$$ 
(Softmax policy)

where $\large \phi(s, a)$ are feature vectors and $\large \theta$ are learnable parameters.

#### Policy Evaluation

Given a policy $\large \pi$, we can evaluate its performance through its value functions, which we'll explore in the next section.

### Value Functions: Quantifying Goodness

#### State-Value Functions

The **state-value function** $\large V^\pi(s)$ gives the expected return when starting in state $\large s$ and following policy $\large \pi$:

$$\large 
V^\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \mid S_t = s\right]
$$

#### Action-Value Functions

The **action-value function** $\large Q^\pi(s, a)$ gives the expected return when starting in state $\large s$, taking action $\large a$, and then following policy $\large \pi$:

$$\large 
Q^\pi(s, a) = \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \mid S_t = s, A_t = a\right]
$$

#### Bellman Equations

The Bellman equations express the recursive relationship between value functions:

**Bellman Expectation Equation for $\large V^\pi$**: 

$$\large 
V^\pi(s) = \sum_{a} \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a)[r + \gamma V^\pi(s')]
$$

**Bellman Expectation Equation for $\large Q^\pi$**: 

$$\large 
Q^\pi(s, a) = \sum_{s', r} p(s', r \mid s, a)\left[r + \gamma \sum_{a'} \pi(a' \mid s') Q^\pi(s', a')\right]
$$

These equations form the basis for many RL algorithms, including temporal difference learning and Q-learning.

#### Optimal Value Functions

The **optimal state-value function** $\large V^_(s)$ is the maximum value achievable from state $\large s$: 

$$\large 
V^_(s) = \max_\pi V^\pi(s)
$$

The **optimal action-value function** $\large Q^_(s, a)$ is the maximum value achievable from state $\large s$ taking action $\large a$: 

$$\large 
Q^_(s, a) = \max_\pi Q^\pi(s, a)
$$

**Bellman Optimality Equations**: 

$$\large 
V^_(s) = \max_a \sum_{s', r} p(s', r \mid s, a)[r + \gamma V^_(s')]
$$ 

$$\large 
Q^_(s, a) = \sum_{s', r} p(s', r \mid s, a)\left[r + \gamma \max_{a'} Q^_(s', a')\right]
$$

### Policy Gradient Methods: Direct Optimization

#### Motivation and Advantages

Policy gradient methods directly optimize the policy parameters $\theta$ to maximize expected return, offering several advantages:

1.  **Direct Optimization**: No need to compute value functions explicitly
2.  **Continuous Actions**: Natural handling of continuous action spaces
3.  **Stochastic Policies**: Can learn inherently stochastic optimal policies
4.  **Convergence Guarantees**: Under certain conditions, guaranteed to converge to local optima

#### Policy Gradient Theorem

The policy gradient theorem provides the foundation for policy gradient methods. It states that the gradient of the expected return with respect to policy parameters is:

$$\large 
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(A_t \mid S_t) G_t\right]
$$

where:

-   $\large J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$ is the objective function
-   $\large \tau$ is a trajectory sampled from policy $\large \pi_\theta$
-   $\large G_t$ is the return from time step $\large t$

#### Proof Sketch of Policy Gradient Theorem

The proof involves several key steps:

1.  **Express the objective function**: 

$$\large 
J(\theta) = \sum_\tau P(\tau \mid \theta) R(\tau)
$$
    
2.  **Take the gradient**: 

$$\large 
\nabla_\theta J(\theta) = \sum_\tau \nabla_\theta P(\tau \mid \theta) R(\tau)
$$
    
3.  **Use the log-derivative trick**: 

$$\large 
\nabla_\theta P(\tau \mid \theta) = P(\tau \mid \theta) \nabla_\theta \log P(\tau \mid \theta)
$$
    
4.  **Express trajectory probability**: 

$$\large 
P(\tau \mid \theta) = \rho_0(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t \mid s_t) p(s_{t+1} \mid s_t, a_t)
$$
    
5.  **Simplify the gradient**: 

$$\large 
\nabla_\theta \log P(\tau \mid \theta) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t)
$$
    
6.  **Convert back to expectation**: 

$$\large 
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(A_t \mid S_t) R(\tau)\right]
$$

### Gradient Ascent in Policy Space

With the policy gradient theorem, we can perform gradient ascent:

$$\large 
\theta_{t+1} = \theta_t + \alpha \nabla_\theta J(\theta_t)
$$

where $\large \alpha$ is the learning rate.

### REINFORCE Algorithm: Monte Carlo Policy Gradient

#### Algorithm Derivation

REINFORCE (REward Increment = Nonnegative Factor × Offset Reinforcement × Characteristic Eligibility) is the simplest policy gradient algorithm. It uses the policy gradient theorem directly with Monte Carlo sampling.

The key insight is to replace the return $\large R(\tau)$ in the policy gradient theorem with the actual return $\large G_t$ from each time step:

$$\large 
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=0}^{T_i-1} \nabla_\theta \log \pi_\theta(A_t^{(i)} \mid S_t^{(i)}) G_t^{(i)}
$$

#### Implementation Details

**Loss Function**: Since most optimizers perform gradient descent (minimization), we define: 

$$\large 
L(\theta) = -\frac{1}{N} \sum_{i=1}^{N} \sum_{t=0}^{T_i-1} \log \pi_\theta(A_t^{(i)} \mid S_t^{(i)}) G_t^{(i)}
$$

**Return Calculation**: For each time step $t$ in episode $\large i$: 

$$\large 
G_t^{(i)} = \sum_{k=t}^{T_i-1} \gamma^{k-t} R_{k+1}^{(i)}
$$

#### Variance Reduction Techniques

**Baseline Subtraction**: Subtract a state-dependent baseline $\large b(s)$ to reduce variance: 

$$\large 
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(A_t \mid S_t) (G_t - b(S_t))\right]
$$

Common baselines include:

-   Moving average of returns
-   State-value function $\large V^\pi(s)$ (leading to Actor-Critic methods)

**Causality**: Only use future rewards for each action: 

$$\large 
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(A_t \mid S_t) \sum_{k=t}^{T-1} \gamma^{k-t} R_{k+1}\right]
$$