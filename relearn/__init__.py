"""A reinforcement learning (RL) package named relearn.

This package provides a framework for simulating and experimenting with
various reinforcement learning scenarios. It includes foundational classes for
creating Markov Decision Processes (MDPs), defining environments, agents,
actions, states, and rewards, as well as implementing policies and learning algorithms.

Modules:
- environment: Defines the RL environment, including actions, states, rewards,
  and transitions.
- agent: Contains classes and methods for RL agents, including policy definitions
  and decision-making processes.
- mdp: Implements the Markov Decision Process framework, encapsulating the
  interaction between agents and environments over time.

Designed for educational purposes, research, and practical application, relearn
aims to facilitate the exploration of reinforcement learning theories and practices.
"""

import relearn.agent
import relearn.environment
import relearn.mdp

__all__ = ["environment", "agent", "mdp"]
__version__ = "0.1.2"
