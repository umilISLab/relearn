"""Implements reinforcement learning agents and related concepts.

Focuses on the definition of RL agents, encompassing the policy under which an agent
operates, and the decision-making process. The module allows for the creation of
custom policies, enabling experimentation with different strategies and behaviors
within an RL environment.
"""

from random import choices
from numpy import array, random
from relearn.environment import State, Action


class Policy:
    """Defines a policy under which an RL agent operates, mapping states to actions.

    Attributes:
        probabilities (array): A matrix representing the probability mass function of the policy.

    Args:
        probabilities (array): State-action probabilities matrix defining the policy.
    """

    def __init__(self, probabilities: array):
        self.state_action_probas = probabilities

    def pmf(self, action: Action, state: State) -> float:
        """Probability mass function of policy pi. Computes the probability of
        an action a given a state s, pi(a|s).

        Args:
            action (Action): a, action to be evaluated
            state (State): s, starting state to select feasible actions

        Returns:
            float: pi(a|s), probability value of an action to occure given a
            state
        """
        return self.state_action_probas[state.idx, action.idx]

    def select_action(self, state: State) -> Action:
        """Selects an action depending on
        the selected state, following a policy.

        Args:
            state (State): current state

        Returns:
            Action: chosen action, given the policy
        """
        if state.end_state:
            raise ValueError("End states have not feasible actions.")
        selected_action = choices(
            state.actions,
            [self.pmf(action, state) for action in state.actions],
            k=1,
        )[0]
        return selected_action

    @staticmethod
    def random_policy(n_states: int, n_actions: int, epsilon=None) -> object:
        """Generates a random policy.

        Args:
            n_states (int): number of possible states
            n_actions (int): number of actions to choose from
            epsilon (float, optional): if a soft policy has to be returned, set
            epsilon to a relatively small float. Defaults to None.

        Returns:
            Policy: the generated random (soft) policy
        """
        random_array = random.rand(n_states, n_actions)
        if epsilon:
            thresh = epsilon / n_actions
            random_array[random_array < thresh] = thresh
        random_array = random_array / random_array.sum(axis=1, keepdims=True)
        return Policy(probabilities=random_array)


class Agent:
    """Represents a reinforcement learning agent capable of making decisions within an environment.

    Attributes:
        policy (Policy): The policy according to which the agent makes decisions.

    Args:
        policy (Policy): An instance of Policy that guides the agent's decisions.
    """

    def __init__(self, policy: Policy):
        self.policy = policy

    def make_choice(self, state: State) -> Action:
        """Shortcut for Policy's select_action().
        Function implementing the decision process given a state

        Args:
            state (State): current state

        Returns:
            Action: chosen action
        """
        selected_action = self.policy.select_action(state)
        return selected_action
