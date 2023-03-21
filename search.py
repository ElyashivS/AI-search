# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    from util import Stack

    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** MY CODE HERE ***"

    # I COMMENTED ONLY THE DFS FUNCTION. THE OTHER FUNCTIONS ARE SIMILAR WITH MINOR CHANGES,
    # SO I DON'T THINK THEY REQUIRE COMMENTS TOO.

    # Initialize variables
    front = Stack()  # DFS implementation requires stack for the front.
    visited = []  # Visited nodes.
    node = problem.getStartState()  # Initialize node state.
    path = []  # Remember the path to the solution.
    cost = 0  # How much is it cost?

    if problem.isGoalState(node):  # Checks if the start state is the goal
        return []

    # Pushes to the front the following fields:
    # 1) Current node.
    # 2) Path until this node.
    # 3) Cost until this node.
    front.push((node, path, cost))

    while not front.isEmpty():  # As long as we haven't finished the maze, do:
        (node, path, cost) = front.pop()  # Pop the next position from the front, and explore it.
        visited.append(node)  # Add the current node to the visited list so we won't check it again.

        if problem.isGoalState(node):  # if finished, return the path.
            return path

        # Explore the successor of the current node,
        # and push it to stack only if it's not in the visited list.
        for successor in problem.getSuccessors(node):
            child = successor[0]
            child_path = path + [successor[1]]
            child_cost = problem.getCostOfActions(child_path)
            if child not in visited:
                front.push((child, child_path, child_cost))

    return []  # If there is no nodes left in the front, we did not find a path to the goal.


def breadthFirstSearch(problem):
    from util import Queue

    """Search the shallowest nodes in the search tree first."""
    "*** MY CODE HERE ***"

    front = Queue()
    visited = []
    node = problem.getStartState()
    path = []
    cost = 0

    if problem.isGoalState(node):
        return []

    front.push((node, path, cost))

    while not front.isEmpty():
        (node, path, cost) = front.pop()
        visited.append(node)

        if problem.isGoalState(node):
            return path

        for successor in problem.getSuccessors(node):
            child = successor[0]
            child_path = path + [successor[1]]
            child_cost = problem.getCostOfActions(child_path)
            if (child not in visited) and (
                    child not in (state[0] for state in front.list)):  # Checks also if already visited
                front.push((child, child_path, child_cost))

    return []


def uniformCostSearch(problem):
    from util import PriorityQueue

    """Search the node of least total cost first."""
    "*** MY CODE HERE ***"

    front = PriorityQueue()  # Implementation of Priority Queue for better paths.
    # For example, maybe we would rather safer and rich in food paths.
    visited = []
    node = problem.getStartState()
    path = []

    if problem.isGoalState(node):
        return []

    front.push((node, path), 0)

    while not front.isEmpty():
        node, path = front.pop()
        visited.append(node)

        if problem.isGoalState(node):
            return path

        for successor in problem.getSuccessors(node):
            child = successor[0]
            child_path = path + [successor[1]]
            child_cost = problem.getCostOfActions(child_path)

            # Checks also if the node isn't in the Queue
            if (child not in visited) and (child not in (state[2][0] for state in front.heap)):
                front.push((child, child_path), child_cost)

            # If the node is in the queue (and hasn't explored), update node to the better cost.
            elif (child not in visited) and (child in (state[2][0] for state in front.heap)):
                for state in front.heap:
                    if state[2][0] == child:
                        old_cost = problem.getCostOfActions(state[2][1])

                if old_cost > child_cost:
                    front.update((child, child_path), child_cost)

    return []


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    from util import PriorityQueueWithFunction

    """Search the node that has the lowest combined cost and heuristic first."""
    "*** MY CODE HERE ***"

    front = PriorityQueueWithFunction(lambda x: problem.getCostOfActions(x[1]) + heuristic(x[0], problem))  #
    visited = []
    node = problem.getStartState()
    path = []

    if problem.isGoalState(node):  # Checks if the start state is the goal
        return []

    # Pushes to the front the following fields:
    # 1) Current node.
    # 2) Path until this node.
    # 3) Heuristic function
    front.push((node, path, heuristic))

    while not front.isEmpty():
        (node, path, cost) = front.pop()

        if node in visited:  # Don't recheck nodes
            continue

        visited.append(node)

        if problem.isGoalState(node):
            return path

        for successor in problem.getSuccessors(node):
            child = successor[0]
            child_path = path + [successor[1]]
            if child not in visited:
                front.push((child, child_path, heuristic))

    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
