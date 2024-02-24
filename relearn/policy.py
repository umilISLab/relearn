"""
This module implements the Policy class, which extends the functionality of the ProbabilityArray
from the relearn.utils module. The Policy class is designed to model decision-making policies in
reinforcement learning environments, associating probabilities with actions given certain states.

The Policy class provides a structured way to represent and manipulate policies, making it easier
to calculate the probability mass function (pmf) for actions, select actions based on the current
state, and generate random policies for experimentation or initialization purposes.

Dependencies:
    - numpy for array operations and mathematical computations.
    - relearn.utils for the ProbabilityArray base class, ensuring policy arrays adhere to 
      probability distribution constraints.
    - relearn.environment for State and Action classes, which define the states and actions 
      within the reinforcement learning environment.

Classes:
    Policy: Represents a policy as a probability distribution over actions for each state.
"""

from typing import List
from random import choices
import numpy as np
from relearn.utils import ProbabilityArray
from relearn.environment import State, Action


class Policy(ProbabilityArray):
    """Represents a reinforcement learning policy as a probability distribution over
    actions for each state.

    The Policy class extends ProbabilityArray, ensuring that each row of the array
    represents a valid
    probability distribution across actions for a given state. This model allows
    for the calculation of
    action probabilities and the selection of actions based on these probabilities.

    Attributes:
        actions (List[Action]): A list of Action objects available in the environment.

    Methods:
        pmf(action: Action, state: State) -> float: Returns the probability of
        taking an action given a state.
        select_action(state: State) -> Action: Selects an action based on the
        policy for the given state.
        random_policy(n_states: int, actions: List[Action], epsilon=None) -> any:
        Generates a random policy array.
    """

    # TODO capire come chiamare super e se devo chiamare super
    def __new__(
        cls, input_array, actions: List[Action], allow_zero_vectors: bool = False
    ):
        super().__new__(input_array, allow_zero_vectors)
        obj = np.asarray(input_array).view(cls)
        if len(actions) != obj.shape[-1]:
            raise ValueError("Length of actions must be equal to last array dimension.")
        obj.actions = actions
        return obj

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        if obj is None:
            return
        # pylint: disable=W0201
        self.actions = getattr(obj, "actions", None)

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
        return self[state.idx, action.idx]

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
            self.actions,
            self[state.idx, :],
            k=1,
        )[0]
        return selected_action

    @staticmethod
    def random_policy(n_states: int, actions: List[Action], epsilon=None) -> any:
        """Generates a random policy.

        Args:
            n_states (int): number of possible states
            actions (int): list of actions to choose from
            epsilon (float, optional): if a soft policy has to be returned, set
            epsilon to a relatively small float. Defaults to None.

        Returns:
            Policy: the generated random (soft) policy
        """
        random_array = np.random.rand(n_states, len(actions))
        if epsilon:
            thresh = epsilon / len(actions)
            random_array[random_array < thresh] = thresh
        random_array = random_array / random_array.sum(axis=1, keepdims=True)
        return Policy(random_array, actions=actions)
