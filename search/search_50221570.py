from problem import State, SokobanProblem, ActionType
from pathlib import Path
from typing import List, Tuple, Any
import numpy as np

from queue import PriorityQueue


def state_to_tuple(state: State) -> tuple[
    tuple[tuple, ...], tuple[tuple[tuple[int]], Any, Any, float, int] | None, tuple[str] | None, float, int]:
    """
    Convert a State object to a tuple where the "map" attribute is a tuple of tuples and "parent" attribute is also a tuple
    (recursively). Also, the "action" attribute is a tuple with a single element (the enum member name) instead of an enum.

    Args:
        state: The State object to be converted.

    Returns:
        A tuple representing the State object, where the "map" attribute is a tuple of tuples, "parent" attribute is a tuple
        (recursively), and "action" attribute is a tuple with a single element (the enum member name) or None if the
        original "action" attribute is None.
    """
    # Convert map numpy array to tuple of tuples
    map_tuple = tuple(tuple(row) for row in state.map)
    # Convert parent object to tuple (recursively)
    parent_tuple = state_to_tuple(state.parent) if state.parent else None
    # Convert action object to tuple with single element (name)
    action_tuple = (state.action.name,) if state.action else None
    return map_tuple, parent_tuple, action_tuple, state.reward_so_far, state.depth


def tuple_to_state(t: tuple[
    tuple[tuple, ...], tuple[tuple[tuple[int]], Any, Any, float, int] | None, tuple[str] | None, float, int]) -> State:
    """
    Convert a tuple representing a State object to the original State object, where the "map" attribute is a numpy array
    and "parent" attribute is also a State object (recursively). Also, the "action" attribute is an enum member instead
    of a tuple with a single element (the enum member name).

    Args:
        t: The tuple representing the State object to be converted.

    Returns:
        A State object representing the original tuple, where the "map" attribute is a numpy array, "parent" attribute is
        also a State object (recursively), and "action" attribute is an enum member or None if the original "action"
        attribute is None.
    """
    # Convert map tuple of tuples to numpy array
    map_array = np.array(t[0])
    # Convert parent tuple to State object (recursively)
    parent_obj = tuple_to_state(t[1]) if t[1] else None
    # Convert action tuple to enum member
    action_obj = ActionType[t[2][0]] if t[2] else None
    return State(map=map_array, parent=parent_obj, action=action_obj, reward_so_far=t[3], depth=t[4])


class Assignment1:
    STUDENT_ID = Path(__file__).stem.split('_')[1]

    def search(self, problem: SokobanProblem) -> List[ActionType]:
        """
        - Which algorithm that you designed?
        Uniform Cost Search.

        - Why did you select it?
        UCS is a reasonable choice since the goal state is not known and the search space can be large. It will allow
        the algorithm to efficiently explore the search space while avoiding getting stuck in local maxima. It is also
        optimal since it will always find the path with the lowest cost.

        - What does each command do?
        frontier: Initialize a sorted list to store states that need to be expanded.
        reached: Initialize a dictionary to keep track of states that have been visited before.
        while not frontier.is_empty(): Keep expanding states until the frontier is empty.
        state: Get the next state from the frontier to expand.
        cost: Get the cost of the state.
        if problem.is_goal_state(state): Check if the current state is the goal state.
        return state.get_action_sequence(): If the current state is the goal state, return the list of actions that led
        to it.
        for child, reward in problem.expand(state): Expand the current state to generate its child states.
        action_cost: Design action cost.
        new_cost: Calculate the cost of the child state by summation.
        if child in reached and new_cost >= reached[child]: Check if the child state has been visited before and new
        cost is greater/equal to the last one.
        continue: If the child state has been visited before and the new cost is higher, skip it.
        reached[child] = new_cost: Add the child state to the reached dictionary.
        frontier.push(new_cost, child): Add the child state to the frontier.
        return []: If no goal state was found, return an empty list.
        """
        # Initialize a sorted list to store states that need to be expanded.
        frontier = PriorityQueue()
        frontier.put((0, state_to_tuple(problem.initial_state)))

        # Initialize a MyDict object to keep track of states that have been visited before.
        reached = {state_to_tuple(problem.initial_state): 0}

        # Keep expanding states until the frontier is empty.
        while not frontier.empty():
            # Get the next state from the frontier to expand.
            cost, tuple_state = frontier.get()
            state = tuple_to_state(tuple_state)

            # Check if the current state is the goal state.
            if problem.is_goal_state(state):
                # If the current state is the goal state, return the list of actions that led to it.
                return state.get_action_sequence()

            # Expand the current state to generate its child states.
            for child, reward in problem.expand(state):

                # Design action cost by subtracting 10 and calculating the absolute value.
                action_cost = 10 - reward

                # Calculate the cost of the child state.
                new_cost = cost + action_cost

                # Check if the child state has been visited before.
                tuple_child = state_to_tuple(child)
                if tuple_child in reached and new_cost >= reached[tuple_child]:
                    # If the child state has been visited before and the new cost is higher, skip it.
                    continue

                # Add the child state to the reached dictionary.
                reached[tuple_child] = new_cost

                frontier.put((new_cost, tuple_child))

        # If no goal state was found, return an empty list.
        return []
