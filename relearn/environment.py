"""Defines the foundational components of a reinforcement learning environment.

This module includes classes that represent actions, states, rewards, and transitions
within the environment and the environment itself, forming the basic building blocks 
for creating RL scenarios.
Additionally, it provides utilities for initializing the environment's dynamics based
on specified transitions, allowing for the simulation and analysis of RL algorithms.
"""

import numpy as np


class Action:
    """Represents an action within a reinforcement learning environment.

    Attributes:
        name (str): The unique identifier for the action.

    Args:
        name (str): The name of the action.
    """

    def __init__(self, name: str):
        self.name = name


class State:
    """Defines a state in the reinforcement learning environment.

    Attributes:
        name (str): The name identifying the state
        end_state (bool): Indicates if this state is an end state.
        start_state (bool): Indicates if this state is a start state.

    Args:
        name (str): The name of the state.
        is_end_state (bool, optional): Marks the state as an end state. Default is False.
        is_start_state (bool, optional): Marks the state as a start state. Default is False.
    """

    def __init__(
        self,
        name: str,
        is_end_state: bool = False,
        is_start_state: bool = False,
    ):
        self.name = name
        self.end_state = is_end_state
        self.start_state = is_start_state


class Reward:
    """Encapsulates the concept of a reward in an RL environment.

    Attributes:
        value (float): The numerical value of the reward.
        initial_reward (bool): Indicates if this is an initial reward.

    Args:
        value (float): The value of the reward.
        initial_reward (bool, optional): Marks the reward as initial. Default is False.
    """

    def __init__(self, value: float, initial_reward: bool = False):
        self.value = value
        self.initial_reward = initial_reward


class Transition:
    """Defines a transition in the RL environment, from one state to another via an action.

    Attributes:
        start_state_name (str): The name of the starting state.
        action_name (str): The name of the action leading to the transition.
        landing_state_name (str): The name of the resulting state after the action.
        reward_value (float): The reward received after the transition.
        probability (float): The probability of the transition occurring.

    Args:
        start_state_name (str): Name of the starting state.
        action_name (str): Name of the action.
        landing_state_name (str): Name of the resulting state.
        reward_value (float): Reward value for the transition.
        probability (float): Probability of the transition.

    Raises:
        ValueError: If the probability is not within the (0,1] range.
    """

    def __init__(
        self,
        start_state_name: str,
        action_name: str,
        landing_state_name: str,
        reward_value: float,
        probability: float,
    ):
        if probability <= 0 or probability > 1:
            raise ValueError("Probability must be in (0,1]")
        self.start_state = start_state_name
        self.action = action_name
        self.end_state = landing_state_name
        self.reward = reward_value
        self.probability = probability


def initialise_dynamics_from_quadruples(
    transitions_list: list[Transition],
    states: list[State],
    actions: list[Action],
    rewards: list[Reward],
):
    """Composes the dynamics array starting from the definition of the single
    transitions.

    Args:
        transitions_list (list[Transition]): list of transitions to populate
        the environment's dynamics

    Raises:
        ValueError: if cumulative probability of each transition exceeds 1

    Returns:
        np.array: environment's dynamics, where value = 0 in non-feasible
        transitions
    """
    dynamics = np.zeros((len(states), len(actions), len(states), len(rewards)))
    for transition in transitions_list:
        start_state_idx = [
            state.idx for state in states if state.name == transition.start_state
        ][0]
        end_state_idx = [
            state.idx for state in states if state.name == transition.end_state
        ][0]
        action_idx = [
            action.idx for action in actions if action.name == transition.action
        ][0]
        reward_idx = [
            reward.idx for reward in rewards if reward.value == transition.reward
        ]
        dynamics[start_state_idx, action_idx, end_state_idx, reward_idx] = (
            transition.probability
        )
    if (np.sum(dynamics, axis=(2, 3)) > 1).any():
        raise ValueError("Transition probabilities must sum to 1")

    return dynamics


class Environment:
    """Defines the structure and dynamics of a reinforcement learning environment.

    This class encapsulates the entire RL environment, including the states, actions,
    rewards, and transitions that define how agents interact with the environment.
    It provides the foundation for simulating RL scenarios, allowing for the
    implementation and testing of various RL algorithms.

    Attributes:
        states (list[State]): A list of the states in the environment.
        actions (list[Action]): A list of the actions possible in the environment.
        rewards (list[Reward]): A list of rewards that can be obtained in the environment.
        dynamics (np.array): A 4D array representing the transition probabilities
            between states given actions, and the associated rewards.
        initial_state (State): The starting state of the environment for the agent.
        initial_reward (Reward, optional): An initial reward, typically set to zero,
            used to start reward accumulations.

    Args:
        states (list[State]): The states comprising the environment.
        actions (list[Action]): The actions possible within the environment.
        rewards (list[Reward]): The rewards that can be obtained.
        transitions (list[Transition]): The transitions defining
            state-action-reward-state dynamics.

    Raises:
        ValueError: If any of the states, actions, or rewards lists contain
            non-unique elements based on their names or values.
    """

    def __init__(
        self,
        states: list[State],
        actions: list[Action],
        rewards: list[Reward],
        transitions: list[Transition],
    ):
        if len(states) != len(set([state.name for state in states])):
            raise ValueError("States names must be unique")
        if len(actions) != len(set([action.name for action in actions])):
            raise ValueError("Actions names must be unique")
        if len(rewards) != len(set([reward.value for reward in rewards])):
            raise ValueError("Rewards values must be unique")

        # enumerating actions, rewards and states
        for idx, action in enumerate(actions):
            action.idx = idx
        self.actions = actions

        self.initial_reward = None
        for idx, reward in enumerate(rewards):
            reward.idx = idx
            if reward.value == 0:
                reward.initial_reward = True
                self.initial_reward = reward
        self.rewards = rewards
        if not self.initial_reward:
            self.rewards[len(self.rewards)] = Reward(0, True)

        self.initial_state = None
        for idx, state in enumerate(states):
            state.idx = idx
            if state.start_state:
                self.initial_state = state
        self.states = states
        if not self.initial_state:
            self.initial_state = self.states[0]

        self.dynamics = initialise_dynamics_from_quadruples(
            transitions, self.states, self.actions, self.rewards
        )

        for state in self.states:
            state.actions = [
                self.actions[a]
                for a in np.unique(np.argwhere(self.dynamics[state.idx])[:, 0])
            ]

    def state_reward_proba(
        self, next_state: State, reward: Reward, state: State, action: Action
    ) -> float:
        """returns the probability of having a state and a reward given the
        current state and the chosen action

        Args:
            next_state (State): state s', the state to be evaluated after the
                action a
            reward (Reward): reward obtained if in state s'
            state (State): state s, the current state
            action (Action): action a, the chosen action from the state s

        Returns:
            float: p(s', r |s, a). Dynamics of the MDP; That is, for particular
            state s' and reward r, retrieve the  probability of those values
            occurring at time t, given particular values of the preceding state
            s and action a
        """
        return self.dynamics[state.idx, action.idx, next_state.idx, reward.idx]

    def state_transition_proba(
        self, next_state: State, state: State, action: Action
    ) -> float:
        """computes transition probabilities

        Args:
            next_state (State): state s', next state of interest for the
                computation of the probability
            state (State): state s, current state
            action (Action): action a, action chosen from current state

        Returns:
            float: p(s' |s, a). Probability of landing in state s' given
            current state s and action a
        """
        return np.sum(self.dynamics[state.idx, action.idx, next_state.idx, :])

    def state_action_function(self, state: State, action: Action) -> float:
        """returns expected rewards given a state-action pair

        Args:
            state (State): state s, current state
            action (Action): action a, action chosen from current state

        Returns:
            float: r(s, a). Expected reward for state-action pair
        """
        return np.sum(
            reward.value * np.sum(self.dynamics[state.idx, action.idx, :, reward.idx])
            for reward in self.rewards
        )

    def state_action_state_function(
        self, state: State, action: Action, next_state: State
    ) -> float:
        """returns expected rewards given a state-action-nextstate triple

        Args:
            state (State): state s, current state
            action (Action): action a, action chosen from current state
            next_state (State): state s', next state after performing action a

        Returns:
            float: r(s, a, s'). Expected rewards for state–action–next-state
            triples
        """

        return np.sum(
            reward.value
            * self.dynamics[state.idx, action.idx, next_state.idx, reward.idx]
            for reward in self.rewards
        )

    def behave(self, state: State, action: Action) -> (State, float):
        """retrieve the feedback of the current environment given a current
        state and the chosen action

        Args:
            state (State): current state
            action (Action): chosen action

        Returns:
            State: next state
            float: reward
        """
        probabilities = self.dynamics[state.idx, action.idx, :, :]
        # blocking non feasible actions
        if np.sum(probabilities) > 0:
            # sampling index from the 2D probabilities array
            i = np.random.choice(np.arange(probabilities.size), p=probabilities.ravel())
            # pylint: disable-next=unbalanced-tuple-unpacking
            next_state_idx, next_reward_idx = np.unravel_index(i, probabilities.shape)
            return self.states[next_state_idx], self.rewards[next_reward_idx]
        else:
            raise ValueError("Selected action is not feasible in selected state")
