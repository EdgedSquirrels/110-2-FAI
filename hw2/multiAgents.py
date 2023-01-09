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


from cmath import inf
import enum
from json.encoder import INFINITY

from numpy import Infinity, average
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
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        "Add more of your code here if you want to"

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
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        h, w = newFood.height, newFood.width
        newFoodPositions = [(i,j) for i in range(w) for j in range(h) if newFood[i][j]]

        newGhostStates = successorGameState.getGhostStates()
        # print(newGhostStates)

        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        newGhostPositions = [ghostState.getPosition() for ghostState in newGhostStates]
        
        score = successorGameState.getScore()
        for i, newGhostPosition in enumerate(newGhostPositions):
            if newScaredTimes[i] <= 2:
                if manhattanDistance(newPos, newGhostPosition) <= 2:
                    score -= 100
                if manhattanDistance(newPos, newGhostPosition) <= 1:
                    score -= 100
        
        closestFoodDist = INFINITY

        if len(newFoodPositions):
            for newFoodPosition in newFoodPositions:
                closestFoodDist = min(closestFoodDist, manhattanDistance(newFoodPosition, newPos))
        else:
            closestFoodDist = 0
        score -= closestFoodDist * 0.5

        """
        print()
        print(" Action:", action)
        print(" newPos:", newPos)
        print(" newFood:", newFood)
        print(" newFoodPostions:", newFoodPositions)
        print(" newScaredTimes:", newScaredTimes)
        print(" newGhostPositions:", newGhostPositions)
        print(" newScore:", successorGameState.getScore())
        print("capsules:", successorGameState.getCapsules())
        newFoodPositions.extend(successorGameState.getCapsules().copy())
        """

        "*** YOUR CODE HERE ***"
        return score

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

    def minimax(self, currentGameState: GameState, currentDepth, agentIndex, action):
        if currentDepth >= self.depth or currentGameState.isLose() or currentGameState.isWin():
            return self.evaluationFunction(currentGameState)
    
        currentGameState = currentGameState.generateSuccessor(agentIndex, action)

        if currentGameState.isLose() or currentGameState.isWin():
            return self.evaluationFunction(currentGameState)

        # determine who should do the next action
        agentIndex += 1
        if agentIndex >= currentGameState.getNumAgents():
            agentIndex = 0
            currentDepth += 1
        if currentDepth >= self.depth:
            return self.evaluationFunction(currentGameState)
        
        # print(currentDepth, agentIndex, action)

        legalMoves = currentGameState.getLegalActions(agentIndex)
        scores = [self.minimax(currentGameState, currentDepth, agentIndex, act) for act in legalMoves]

        return max(scores) if agentIndex == 0 else min(scores)


    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

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

        "*** YOUR CODE HERE ***"

        legalMoves = gameState.getLegalActions()

        scores = [self.minimax(gameState, 0, 0, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        # chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        
        return legalMoves[bestIndices[0]]
        # return legalMoves[chosenIndex]

        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def alphaBeta(self, currentGameState: GameState, currentDepth, agentIndex, action, a = -INFINITY, b = INFINITY):
        if currentDepth >= self.depth or currentGameState.isLose() or currentGameState.isWin():
            return self.evaluationFunction(currentGameState)
    
        currentGameState = currentGameState.generateSuccessor(agentIndex, action)

        if currentGameState.isLose() or currentGameState.isWin():
            return self.evaluationFunction(currentGameState)

        # determine who should do the next action
        agentIndex += 1
        if agentIndex >= currentGameState.getNumAgents():
            agentIndex = 0
            currentDepth += 1
        if currentDepth >= self.depth:
            return self.evaluationFunction(currentGameState)
        
        # print(currentDepth, agentIndex, action)

        legalMoves = currentGameState.getLegalActions(agentIndex)
        v = -INFINITY if agentIndex == 0 else INFINITY
        if agentIndex == 0:
            # get max value
            for act in legalMoves:
                v = max(v, self.alphaBeta(currentGameState, currentDepth, agentIndex, act, a, b))
                a = max(a, v)
                if a > b:
                    break
        else:
            # get min value
            for act in legalMoves:
                v = min(v, self.alphaBeta(currentGameState, currentDepth, agentIndex, act, a, b))
                b = min(b, v)
                if a > b:
                    break
        return v

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        a, v, b = -INFINITY, -INFINITY, INFINITY
        legalMoves = gameState.getLegalActions()
        bestIndex = 0

        for i, act in enumerate(legalMoves):
            v_prev = v
            v = max(v, self.alphaBeta(gameState, 0, 0, act, a, b))
            if v > v_prev:
                bestIndex = i
            a = max(a, v)
            if v > b:
                break

        return legalMoves[bestIndex]


        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    

    def expectimax(self, currentGameState: GameState, currentDepth, agentIndex, action):
        if currentDepth >= self.depth or currentGameState.isLose() or currentGameState.isWin():
            ####
            # return betterEvaluationFunction(currentGameState)
            return self.evaluationFunction(currentGameState)
    
        currentGameState = currentGameState.generateSuccessor(agentIndex, action)

        if currentGameState.isLose() or currentGameState.isWin():
            return self.evaluationFunction(currentGameState)

        # determine who should do the next action
        agentIndex += 1
        if agentIndex >= currentGameState.getNumAgents():
            agentIndex = 0
            currentDepth += 1
        if currentDepth >= self.depth:
            return self.evaluationFunction(currentGameState)
        
        # print(currentDepth, agentIndex, action)

        legalMoves = currentGameState.getLegalActions(agentIndex)
        scores = [self.expectimax(currentGameState, currentDepth, agentIndex, act) for act in legalMoves]

        return max(scores) if agentIndex == 0 else sum(scores) / len(scores)

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        ####
        self.depth = 2
        legalMoves = gameState.getLegalActions()

        scores = [self.expectimax(gameState, 0, 0, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        
        return legalMoves[chosenIndex]

        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    gameState = currentGameState
    pacmanPosition = gameState.getPacmanPosition()
    food = gameState.getFood()
    h, w = food.height, food.width
    foodPositions = [(i,j) for i in range(w) for j in range(h) if food[i][j]]

    ghostStates = gameState.getGhostStates()
    # print(newGhostStates)

    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    ghostPositions = [ghostState.getPosition() for ghostState in ghostStates]
        
    score = gameState.getScore()

    closestTargetDist = INFINITY
    palletPositions = currentGameState.getCapsules()
    targets, targets2 = foodPositions.copy(), []
    targets.extend(palletPositions)

    for i, ghostPosition in enumerate(ghostPositions):
        if scaredTimes[i] <= 2:
            aaaaa = 1
            # if manhattanDistance(pacmanPosition, ghostPosition) <= 2:
                # score -= 100
            if manhattanDistance(pacmanPosition, ghostPosition) <= 1:
                score -= 100
        else:
            score += scaredTimes[i] / 2
            # score -= len(palletPositions) * 50
            if manhattanDistance(pacmanPosition, ghostPosition) <= 1.5 * scaredTimes[i]:
                targets2.append(ghostPosition)
        

    if len(targets) or len(targets2):
        for target in targets2:
            closestTargetDist = min(closestTargetDist, manhattanDistance(target, pacmanPosition)*0.5)
        # if targets2 == []:
        for target in targets:
            closestTargetDist = min(closestTargetDist, manhattanDistance(target, pacmanPosition))
        
    else:
        closestTargetDist = 0
    score -= closestTargetDist * 0.5
    return score
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
