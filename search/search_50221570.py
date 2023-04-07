from problem import State, SokobanProblem, ActionType
from pathlib import Path
from typing import List, Tuple


class PriorityQueue:
    def __init__(self):
        self.queue = []

    def push(self, priority, item):
        """
        Adds a new item to the queue with a given priority
        """
        self.queue.append((priority, item))

    def pop(self):
        """
        Removes and returns the item with the least priority
        """
        if not self.queue:
            return None
        else:
            least_priority = self.queue[0][0]
            least_priority_idx = 0
            for i, (priority, item) in enumerate(self.queue):
                if priority < least_priority:
                    least_priority = priority
                    least_priority_idx = i
            return self.queue.pop(least_priority_idx)  # Return the item and priority

    def is_empty(self):
        """
        Returns True if the queue is empty, False otherwise
        """
        return not self.queue

    def size(self):
        """
        Returns the number of items in the queue
        """
        return len(self.queue)


class MyDict:
    def __init__(self):
        self.keys = []
        self.values = []

    def __getitem__(self, key):
        """
        Return the value associated with the given key if it exists in the keys list,
        or raise a KeyError if it does not.
        """
        try:
            index = self.keys.index(key)
            return self.values[index]
        except ValueError:
            raise KeyError(key)

    def __setitem__(self, key, value):
        """
        Set the value associated with the given key. If the key already exists in the keys
        list, its value is updated. If not, the key and value are added to the end of the
        keys and values lists, respectively.
        """
        try:
            index = self.keys.index(key)
        except ValueError:
            index = len(self.keys)
            self.keys.append(key)
            self.values.append(value)
        else:
            self.values[index] = value

    def __delitem__(self, key):
        """
        Remove the given key and its associated value from the MyDict object if it exists in
        the keys list, or raise a KeyError if it does not.
        """
        try:
            index = self.keys.index(key)
        except ValueError:
            raise KeyError(key)
        else:
            del self.keys[index]
            del self.values[index]

    def __contains__(self, key):
        """
        Return True if the given key exists in the keys list, and False otherwise.
        """
        return key in self.keys

    def __len__(self):
        """
        Return the number of keys in the MyDict object.
        """
        return len(self.keys)

    def __repr__(self):
        """
        Return a string representation of the MyDict object in the form of
        MyDict({key1: value1, key2: value2, ...}).
        """
        items = ", ".join([f"{key}: {value}" for key, value in zip(self.keys, self.values)])
        return f"MyDict({{{items}}})"

    def __str__(self):
        """
        Return a string representation of the MyDict object in the form of
        key1: value1 \n key2: value2 \n ...
        """
        items = "\n".join([f"{key}: {value}" for key, value in zip(self.keys, self.values)])
        return f"{items}"


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
        frontier.push(0, problem.initial_state)

        # Initialize a MyDict object to keep track of states that have been visited before.
        reached = MyDict()

        # Add the initial state to the reached dictionary with a value of 0.
        reached[problem.initial_state] = 0

        # Keep expanding states until the frontier is empty.
        while not frontier.is_empty():
            # Get the next state from the frontier to expand.
            cost, state = frontier.pop()

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
                if child in reached and new_cost >= reached[child]:
                    # If the child state has been visited before and the new cost is higher, skip it.
                    continue

                # Add the child state to the reached dictionary.
                reached[child] = new_cost

                frontier.push(new_cost, child)

        # If no goal state was found, return an empty list.
        return []