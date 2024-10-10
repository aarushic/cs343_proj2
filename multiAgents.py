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
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        #evaluation function - typically weighted linear sum of features
        score = successorGameState.getScore()
        minFoodDist = float('inf')
        for food in newFood.asList():
            dist = util.manhattanDistance(newPos, food)
            minFoodDist = min(minFoodDist, dist)
        score += 1/minFoodDist

        ghostPos = successorGameState.getGhostPositions() 
        for ghost in ghostPos:
            dist = util.manhattanDistance(newPos, ghost)
            if dist < 5:
                score -= 1000000
        
        for i, ghostState in enumerate(newGhostStates):
            ghostPos = ghostState.getPosition()
            dist = util.manhattanDistance(newPos, ghostPos)
            #chase if scared
            if newScaredTimes[i] != 0:
                score += 5000/dist
            
            if newScaredTimes[i] == 0 and dist < 5:
                #bad if close and not scared
                score -= 1000000  

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
        #depth 0
        maxScore = float('-inf')
        bestOutcome = None
        for action in gameState.getLegalActions(0):
            #agentIndex = 0 means Pacman, ghosts are >= 1
            successorState = gameState.generateSuccessor(0, action)
            #ghosts turn
            score = self.minimax(1, 0, successorState)
            if score > maxScore:
                bestOutcome = action
            maxScore = max(score, maxScore)

        return bestOutcome

    def minimax(self, agentIndex, depth, state):
        #evaluate game state to assign score to leaf
        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        #pacman 
        if agentIndex == 0: 
            return self.pacmanMaxValue(depth, state)
        #ghost
        else:
            return self.ghostMinValue(agentIndex, depth, state)
                
    #pacman - maximizer, ghost - minimizer
    def pacmanMaxValue(self, depth, state: GameState):
        v = float('-inf')
        for action in state.getLegalActions(0):
            successorState = state.generateSuccessor(0, action)
            #call minimax for ghost
            v = max(v, self.minimax(1, depth, successorState)) 
        return v

    def ghostMinValue(self, agentIndex, depth, state: GameState):
        v = float('inf')
        nextAgent = agentIndex + 1
        #check if its the last ghost
        if nextAgent >= state.getNumAgents():
            nextAgent = 0
       
        for action in state.getLegalActions(agentIndex):
            successorState = state.generateSuccessor(agentIndex, action)
            if nextAgent == 0:  
                #finish pacman and ghost turn so depth increases
                v = min(v, self.minimax(0, depth + 1, successorState))
            else: 
                #next ghost's turn
                v = min(v, self.minimax(nextAgent, depth, successorState))

        return v
       

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #best score found so far
        alpha = float('-inf')
        #worst score
        beta = float('inf')
        bestOutcome = None

        for action in gameState.getLegalActions(0):
            successorState = gameState.generateSuccessor(0, action)
            score = self.alphaBeta(successorState, 1, 0, alpha, beta)
            if score > alpha:
                bestOutcome = action
                alpha = score
        return bestOutcome
    
    def alphaBeta(self, state, agentIndex, depth, alpha, beta):
        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        if agentIndex == 0:
            return self.pacmanMaxValue(depth, state, alpha, beta)
        else:
            return self.ghostMinValue(agentIndex, depth, state, alpha, beta)

    #pacman - maximizer, ghost - minimizer
    def pacmanMaxValue(self, depth, state: GameState, alpha, beta):
        v = float('-inf')
        for action in state.getLegalActions(0):
            successorState = state.generateSuccessor(0, action)
            v = max(v, self.alphaBeta(successorState, 1, depth, alpha, beta))
            #prune
            if v > beta:
                return v
            alpha = max(alpha, v)

        return v

    def ghostMinValue(self, agentIndex, depth, state: GameState, alpha, beta):
        v = float('inf')
        nextAgent = agentIndex + 1
        if nextAgent >= state.getNumAgents():
            nextAgent = 0
        
        for action in state.getLegalActions(agentIndex):
            successorState = state.generateSuccessor(agentIndex, action)
            if nextAgent == 0:  
                #finish pacman and ghost turn so depth increases
                v = min(v, self.alphaBeta(successorState, 0, depth + 1, alpha, beta))
            else: 
                #next ghost's turn
                v = min(v, self.alphaBeta(successorState, nextAgent, depth, alpha, beta))
        
            #prune
            if v < alpha:
                return v
            beta = min(beta, v)

        return v 
        
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
        "*** YOUR CODE HERE ***"
        maxScore = float('-inf')
        bestOutcome = None

        for action in gameState.getLegalActions(0):
            score = self.expectimax(1, 0, gameState.generateSuccessor(0, action))
            if score > maxScore:
                bestOutcome = action 
                maxScore = max(score, maxScore)
        return bestOutcome

    def expectimax(self, agentIndex, depth, state):
        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        #if next agent if max
        if agentIndex == 0:
            return self.pacmanMaxValue(state, depth)
        #if next agent is expected
        else:
            return self.ghostExpectedValue(agentIndex, depth, state)

    def pacmanMaxValue(self, state, depth):
        v = float('-inf') 
        for action in state.getLegalActions(0):
            v = max(v, self.expectimax(1, depth, state.generateSuccessor(0, action))) 
        
        return v
       
    def ghostExpectedValue(self, agentIndex, depth, state):
        v = 0
        nextAgent = agentIndex + 1
        if nextAgent >= state.getNumAgents(): 
            nextAgent = 0
        
        #choosing uniformly at random from their legal moves
        actions = state.getLegalActions(agentIndex)
        p = 1/len(actions)
        for action in actions:
            successor = state.generateSuccessor(agentIndex, action)
            if nextAgent == 0: 
                v += p * self.expectimax(0, depth + 1, successor)
            else: 
                v += p * self.expectimax(nextAgent, depth, successor)
        
        return v

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    The main factors we chose to focus on were distance to food, number of food pellets left, distance to ghosts, and chasing ghosts when they're scared.
    We reward being close to left over food, having less food left over, being far from non scared ghosts, and being close to scared ghosts. 



    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    score = currentGameState.getScore()
    

   #prefer when pacman is closer to leftover food
    minFoodDist = float('inf')
    for food in newFood.asList():
        currDist = util.manhattanDistance(newPos, food)
        minFoodDist = min(currDist, minFoodDist)
    #extra points for eating food
    if(minFoodDist == 0):
        score += 10
    else:
        score += 1/minFoodDist

    #prefer when more food is eaten
    score -= len(newFood.asList())


    for ghost in newGhostStates:
        ghostPos = ghost.getPosition()
        #penalty for being close to a regular ghost
        if(ghost.scaredTimer == 0):
            currDist = util.manhattanDistance(newPos, ghostPos)
            #this is a loss
            if currDist == 0:
                return float("-inf")
            score -= 1/currDist

    
        #reward for eating a scared ghost
        if(ghost.scaredTimer > 0):
            currDist = util.manhattanDistance(newPos, ghostPos)
            #promotes chasing ghosts until eaten
            if(currDist == 0):
                score += 10
            else:
                score += 1/currDist
    
    return score

# Abbreviation
better = betterEvaluationFunction
