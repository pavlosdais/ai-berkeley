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

from genericpath import exists
from re import S
from typing import Tuple
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
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """Search the deepest nodes in the search tree first."""
    
    # Needed data structures
    st = util.Stack()
    parent = dict()
    steps = list()
    have_visited = set()

    start_state = (problem.getStartState(), None, None)
    st.push(start_state)
    
    while (not st.isEmpty()):
        state = st.pop()
        have_visited.add(state[0])  # Visit the current state

        # Found goal state
        if (problem.isGoalState(state[0])):

            # Trace back the path and reverse it to get the correct sequence of steps
            while (state != start_state):
                steps.append(state[1])
                state = parent[state]
            
            steps.reverse()
            return steps

        # Expand successors
        successors = problem.getSuccessors(state[0])
        for successor in successors:
            if (successor[0] not in have_visited):
                parent[successor] = state
                st.push(successor)
    
    # Solution/Path does not exist
    return None
    

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""

    # Needed data structures
    Q = util.Queue()
    parent = dict()
    steps = list()
    have_visited = set()

    start_state = (problem.getStartState(), None, None)
    have_visited.add(start_state[0])
    parent[start_state] = None
    Q.push(start_state)
    
    while (not Q.isEmpty()):
        state = Q.pop()

        # Found goal state
        if (problem.isGoalState(state[0])):

            # Trace back the path and reverse it to get the correct sequence of steps
            while (state != start_state):
                steps.append(state[1])
                state = parent[state]
            
            steps.reverse()
            return steps

        # Expand successors
        successors = problem.getSuccessors(state[0])
        for successor in successors:
            if (successor[0] not in have_visited):
                Q.push(successor)
                parent[successor] = state
                have_visited.add(successor[0])  # Visit the current state
    
    # Solution/Path does not exist
    return None
    

def uniformCostSearch(problem):
    """Search the node of least total cost first."""

    # Needed data structures
    from util import PriorityQueue
    PQ = util.PriorityQueue()
    have_visited = set()

    PQ.push((problem.getStartState(), list()), 0)

    while(not PQ.isEmpty()):
        state, path = PQ.pop()  # Take current state and path
        have_visited.add(state)  # Visit the current state

        # Found goal state
        if (problem.isGoalState(state)): return path
        
        successors = problem.getSuccessors(state)
        for successor in successors:

            # Already have visited the particular node
            if (successor[0] in have_visited): continue
            
            # Search to see if successor already exists in the frontier(PQ)
            frontier_exists = False
            for element in PQ.heap:
                if (successor[0] == element[2][0]):
                    frontier_exists = True
                    break
            
            new_path = path + [successor[1]]
            new_priority = problem.getCostOfActions(new_path)
            
            # State does not exist either in searched state nor in the frontier, insert it
            if (not frontier_exists):
                PQ.push((successor[0], new_path), new_priority)
            
            # Successor exists in the frontier with a higher path cost - update its path cost
            elif (problem.getCostOfActions(element[2][1]) > new_priority):
                PQ.update((successor[0], new_path), new_priority)
    
    # Solution/Path does not exist
    return None

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    
    # Needed data structures
    from util import PriorityQueue
    PQ = util.PriorityQueue()
    have_visited = set()

    start_state = problem.getStartState()
    PQ.push((start_state, list()), heuristic(start_state, problem))

    while(not PQ.isEmpty()):
        state, path = PQ.pop()  # Take current state and path
        have_visited.add(state)  # Visit the current state

        # Found goal state
        if (problem.isGoalState(state)): return path

        successors = problem.getSuccessors(state)
        for successor in successors:

            # Already have visited the particular node
            if (successor[0] in have_visited): continue
            
            # Search to see if successor already exists in the frontier(PQ)
            frontier_exists = False
            for element in PQ.heap:
                if (successor[0] == element[2][0]):
                    frontier_exists = True
                    break
            
            heuristic_eval = heuristic(successor[0], problem)  # heuristic cost

            new_path = path + [successor[1]]
            new_priority = problem.getCostOfActions(new_path)

            # State does not exist either in searched state nor in the frontier, insert it
            if (not frontier_exists):
                PQ.push((successor[0], new_path), new_priority + heuristic_eval)
            
            # Successor exists in the frontier with a higher path cost - update its path cost
            elif (problem.getCostOfActions(element[2][1]) > new_priority):
                PQ.update((successor[0], new_path), new_priority + heuristic_eval)
    
    # Solution/Path does not exist
    return None


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch