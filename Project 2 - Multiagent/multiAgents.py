# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.
    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.
        getAction chooses among the best options according to the evaluation function.
        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.
        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.
        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)

        # Pac-Man position
        newPos = successorGameState.getPacmanPosition()
        
        # get a list of tuples containing food positions
        newFood = successorGameState.getFood().asList()

        # the number of moves that each ghost will remain scared because of Pacman having eaten a power pellet
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # get distance from ghosts
        ghost_distance = 0
        closest_ghost = 999999
        for ghost in successorGameState.getGhostPositions():
            md = manhattanDistance(newPos, ghost)
            ghost_distance += md
            if (md < closest_ghost): closest_ghost = md
        
        # get distance from foods
        food_distance = 0
        closest_food = 999999
        for food in newFood:
            md = manhattanDistance(newPos, food)
            food_distance += md
            if (md < closest_food): closest_food = md

        # use a bonus score to award/punish certain behaviours

        bonus_score = 0
        
        if (closest_ghost < 3 and closest_ghost > 0): bonus_score = -40/closest_ghost  # punish going near the closest ghost
        elif (closest_food > 0): bonus_score = 10/closest_food  # award going near the closest food

        if action == 'Stop': bonus_score -= 10  # punish standing still

        if (food_distance == 0): return successorGameState.getScore() + bonus_score + sum(newScaredTimes)

        return ghost_distance/(10 * food_distance) + successorGameState.getScore() + sum(newScaredTimes) + bonus_score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.
    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.
    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.
    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    
    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1
        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action
        gameState.getNumAgents():
        Returns the total number of agents in the game
        gameState.isWin():
        Returns whether or not the game state is a winning state
        gameState.isLose():
        Returns whether or not the game state is a losing state
        """

        num_of_agents = gameState.getNumAgents()-1

        def Maximizing(gameState, curr_depth):
            if (curr_depth == 0 or gameState.isWin() or gameState.isLose()):  # base case
                return self.evaluationFunction(gameState)
            
            max_eval = -9999999
            legal_actions = gameState.getLegalActions(0)

            for action in legal_actions:
                new_board = gameState.generateSuccessor(0, action)

                eval = Minimizing(new_board, curr_depth-1, 1)
                if (eval > max_eval): max_eval = eval
                
            return max_eval

        def Minimizing(gameState, curr_depth, agent):
            if (gameState.isWin() or gameState.isLose()):  # base case
                return self.evaluationFunction(gameState)
            
            min_eval = 9999999

            # get legal actions
            legal_actions = gameState.getLegalActions(agent)
            
            for action in legal_actions:
                # play action
                new_board = gameState.generateSuccessor(agent, action)

                if (agent == num_of_agents):
                    eval = Maximizing(new_board, curr_depth)
                else:  # get next ghost
                    eval = Minimizing(new_board, curr_depth, agent+1)  

                if (eval < min_eval): min_eval = eval
                
            return min_eval

        max_eval = -999999999
        best_action = None
        
        search_depth = self.depth-1
        
        legal_actions = gameState.getLegalActions(0)
        for action in legal_actions:
            new_board = gameState.generateSuccessor(0, action)
            
            eval = Minimizing(new_board, search_depth, 1)
            if (eval > max_eval):
                max_eval = eval
                best_action = action

        return best_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        num_of_agents = gameState.getNumAgents()-1

        def Maximizing(gameState, curr_depth, alpha, beta):
            if (curr_depth == 0 or gameState.isWin() or gameState.isLose()):  # base case
                return self.evaluationFunction(gameState)
            
            max_eval = -9999999
            legal_actions = gameState.getLegalActions(0)

            for action in legal_actions:
                # play action
                new_board = gameState.generateSuccessor(0, action)

                eval = Minimizing(new_board, curr_depth-1, 1, alpha, beta)

                if (eval > max_eval): max_eval = eval

                if (max_eval > beta): return eval  # alpha-beta cut off
                if (max_eval > alpha): alpha = max_eval
                
            return max_eval

        def Minimizing(gameState, curr_depth, agent, alpha, beta):
            if (gameState.isWin() or gameState.isLose()):  # base case - stop searching
                return self.evaluationFunction(gameState)
            
            min_eval = 9999999

            # get legal actions
            legal_actions = gameState.getLegalActions(agent)

            for action in legal_actions:
                # play action
                new_board = gameState.generateSuccessor(agent, action)

                if (agent == num_of_agents):
                    eval = Maximizing(new_board, curr_depth, alpha, beta)
                else:  # get next ghost
                    eval = Minimizing(new_board, curr_depth, agent+1, alpha, beta)  

                if (eval < min_eval): min_eval = eval

                if (min_eval < alpha): return eval  # alpha-beta cut off
                if (min_eval < beta): beta = min_eval
                
            return min_eval

        search_depth = self.depth-1
        best_action = None

        max_eval = -999999999
        a = -999999

        legal_actions = gameState.getLegalActions(0)
        for action in legal_actions:

            eval = Minimizing(gameState.generateSuccessor(0, action), search_depth, 1, a, 999999)
            if (eval > max_eval):
                max_eval = eval
                best_action = action
            
            if (a < eval): a = eval

        return best_action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction
        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        num_of_agents = gameState.getNumAgents()-1

        def Maximizing(gameState, curr_depth):
            if (curr_depth == 0 or gameState.isWin() or gameState.isLose()):  # base case
                return self.evaluationFunction(gameState)
            
            max_eval = -99999999
            legal_actions = gameState.getLegalActions(0)

            for action in legal_actions:
                new_board = gameState.generateSuccessor(0, action)

                eval = Minimizing(new_board, curr_depth-1, 1)
                if (eval > max_eval): max_eval = eval
                
            return max_eval

        def Minimizing(gameState, curr_depth, agent):
            if (gameState.isWin() or gameState.isLose()):  # base case
                return self.evaluationFunction(gameState)
            
            total_eval = 0
            num_of_actions = 0

            # get legal actions
            legal_actions = gameState.getLegalActions(agent)
            
            for action in legal_actions:
                # play action
                new_board = gameState.generateSuccessor(agent, action)

                if (agent == num_of_agents):
                    eval = Maximizing(new_board, curr_depth)
                else:  # get next ghost
                    eval = Minimizing(new_board, curr_depth, agent+1)  

                total_eval += eval
                num_of_actions += 1
            
            return total_eval/num_of_actions

        max_eval = -999999999
        best_action = None
        
        search_depth = self.depth-1
        
        legal_actions = gameState.getLegalActions(0)
        for action in legal_actions:
            new_board = gameState.generateSuccessor(0, action)
            # if (new_board.isWin()): return action
            
            eval = Minimizing(new_board, search_depth, 1)

            if (eval > max_eval):
                max_eval = eval
                best_action = action

        return best_action

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).
    DESCRIPTION: <write something here so we know what you did>
    """

    if (currentGameState.isLose()): return -9999999999999
    if (currentGameState.isWin()): return 9999999999999
    
    # Pac-Man position
    newPos = currentGameState.getPacmanPosition()

    # get a list of tuples containing food positions
    newFood = currentGameState.getFood().asList()

    # get a list of tuples containing ghost positions
    ghostPositions = currentGameState.getGhostPositions()

    # the number of moves that each ghost will remain scared because of Pacman having eaten a power pellet
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    nearestGhostDistance = 99999
    ghost_distance = 0
    num_of_ghosts = 0
    for ghost in ghostPositions:
        num_of_ghosts += 1
        md = manhattanDistance(ghost, newPos)
        ghost_distance += md
        if (nearestGhostDistance < md): nearestGhostDistance = md
    
    nearestFoodDistance = 99999
    food_distance = 0
    num_of_foods = 0
    for food in newFood:
        md = manhattanDistance(food, newPos)
        food_distance += md
        if (nearestFoodDistance < md): nearestFoodDistance = md
        num_of_foods += 1
    
    bonusScore = 0
    if (nearestGhostDistance > 0 and nearestGhostDistance < 3):
        bonusScore = -1200/nearestGhostDistance
        if (nearestGhostDistance == 1): bonusScore -= 5000
    elif (nearestFoodDistance > 0):
        bonusScore += 5000/nearestFoodDistance
    
    if (num_of_foods < 4 and food_distance > 0): bonusScore += 5000/food_distance

    if (num_of_ghosts != 0): ghost_score = ghost_distance/(num_of_ghosts*num_of_ghosts)
    else: ghost_score = 800

    capsules = currentGameState.getCapsules()
    num_of_capsules = 0
    capsule_distance = 0
    for capsule in capsules:
        capsule_distance += manhattanDistance(capsule, newPos)
        num_of_capsules += 1
    
    duration = 2*sum(newScaredTimes)
    if (duration == 0):
        duration = 10 * capsule_distance
    
    result = ghost_score - 7*food_distance/num_of_foods + 10*bonusScore + currentGameState.getScore() - duration - 5000*num_of_foods
    return result

# Abbreviation
better = betterEvaluationFunction