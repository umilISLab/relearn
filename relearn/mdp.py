"""Provides an implementation framework for Markov Decision Processes (MDP).

Central to the relearn package, this module encapsulates the essence of reinforcement
learning problems by modeling the interaction between agents and environments over time.
It facilitates the execution of RL algorithms, tracking states, actions, and rewards
to analyze the performance and dynamics of RL strategies.
"""

from itertools import zip_longest
from relearn.environment import Environment, State, Action
from relearn.agent import Agent


class MDP:
    """Models a Markov Decision Process, encapsulating the dynamics of RL problems.

    Attributes:
        agent (Agent): The agent navigating the environment.
        environment (Environment): The environment in which the agent operates.
        time (int): Tracks the current time step within the simulation.
        current_state (State): The current state of the agent within the environment.
        trajectory (dict): Records the states, actions, and rewards experienced by the agent.

    Args:
        agent (Agent): The RL agent involved in the MDP.
        environment (Environment): The RL environment for the MDP.

    Raises:
        ValueError: If the shapes of the policy probabilities and the environment
            dimensions do not match.
    """

    def __init__(self, agent: Agent, environment: Environment):
        if agent.policy.state_action_probas.shape != (
            len(environment.states),
            len(environment.actions),
        ):
            raise ValueError(
                """Probabilities are expected to have shape
                                 num_states x num_actions, with states and
                                 actions defined in MDP"""
            )
        self.agent = agent
        self.environment = environment
        self.time = 0
        self.current_state = self.environment.initial_state
        # (r0, s0, a0),(r1, s1, a1)
        self.trajectory = {
            "rewards": [self.environment.initial_reward],
            "states": [self.environment.initial_state],
            "actions": [],
        }

    def print_trajectory(self):
        """Prints a pretty version of the MDP's trajectory"""
        for traj in zip_longest(
            self.trajectory["rewards"],
            self.trajectory["states"],
            self.trajectory["actions"],
            fillvalue=None,
        ):
            if traj[2]:
                print(
                    f"Reward: {traj[0].value}\tState: {traj[1].name}\t"
                    + f"Action: {traj[2].name}"
                )
            else:
                print(f"""Reward: {traj[0].value}\tState: {traj[1].name}""")

    def iterate(self, starting_state: State = None, starting_action: Action = None):
        """Performs an iteration of the MDP. If starting state and starting
        actions are specified, performs an iteration of the MDP, having
        selected an initial state and action.

        Args:
            starting_state (State, optional): Starting state. Defaults to None.
            starting_action (Action, optional): Starting action. Defaults to None.
        """
        try:
            if starting_state and starting_action:
                self.current_state = starting_state
                # adjust trajectory
                self.trajectory["states"][-1] = starting_state
                self.trajectory["actions"].append(starting_action)
                next_state, reward = self.environment.behave(
                    starting_state, starting_action
                )
            else:
                selected_action = self.agent.make_choice(self.current_state)
                self.trajectory["actions"].append(selected_action)
                next_state, reward = self.environment.behave(
                    self.current_state, selected_action
                )
            self.trajectory["rewards"].append(reward)
            self.trajectory["states"].append(next_state)
            self.current_state = next_state
            self.time += 1
        except ValueError:
            print("End state was met.")
