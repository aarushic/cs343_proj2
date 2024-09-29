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
        ##POTENTIALLY CHANGE SOME LOGIC

        
        #evaluation function - typically weighted linear sum of features
        score = successorGameState.getScore()
        minFoodDist = 1000000
        for food in newFood.asList():
            dist = util.manhattanDistance(newPos, food)
            minFoodDist = min(minFoodDist, dist)
        score += 1/minFoodDist

        ghostPos = successorGameState.getGhostPositions() 
        for ghost in ghostPos:
            dist = util.manhattanDistance(newPos, ghost)
            if dist < 5:
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
        #chatty - maybe change bestAction name
        maxScore = float('-inf')
        bestAction = None
        for action in gameState.getLegalActions(0):
            #chatty??
            #agentIndex = 0 means Pacman, ghosts are >= 1
            successorState = gameState.generateSuccessor(0, action)
            #now ghosts turn
            score = self.minimax(1, 0, successorState)
            if score > maxScore:
                bestAction = action
                maxScore = max(score, maxScore)
        return bestAction

    def minimax(self, agentIndex, depth, state):
        #chatty perhaps

        #reaching max depth or terminal state - evaluate game state to assign score to that leaf
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
        #chatty
        v = float('-inf')
        actions = state.getLegalActions(0)
        if not actions:
            return self.evaluationFunction(state)

        for action in actions:
            successorState = state.generateSuccessor(0, action)
            #call minimax for ghost
            v = max(v, self.minimax(1, depth, successorState)) 

        return v

    def ghostMinValue(self, agentIndex, depth, state: GameState):
        #chattyish
        v = float('inf')
        actions = state.getLegalActions(agentIndex)
        if not actions:
            return self.evaluationFunction(state)
        
        nextAgent = agentIndex + 1
        if nextAgent >= state.getNumAgents():
            #pacman's turn if this was the last ghost
            nextAgent = 0

        for action in actions:
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
        bestAction = None

        for action in gameState.getLegalActions(0):
            successorState = gameState.generateSuccessor(0, action)
            score = self.alphaBeta(successorState, 1, 0, alpha, beta)
            if score > alpha:
                alpha = score
                bestAction = action
        return bestAction
    
    def alphaBeta(self, state, agentIndex, depth, alpha, beta):
        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        if agentIndex == 0:
            return self.pacmanMaxValue(depth, state, alpha, beta)
        else:
            return self.ghostMinValue(agentIndex, depth, state, alpha, beta)

    #pacman - maximizer, ghost - minimizer
    def pacmanMaxValue(self, depth, state: GameState, alpha, beta):
        #chatty
        v = float('-inf')
        actions = state.getLegalActions(0)
        if not actions:
            return self.evaluationFunction(state)

        for action in actions:
            successorState = state.generateSuccessor(0, action)
            nextAgentScore = self.alphaBeta(successorState, 1, depth, alpha, beta)
            v = max(v, nextAgentScore)

            #prune
            if v > beta:
                return v
            alpha = max(alpha, v)
        return v

    def ghostMinValue(self, agentIndex, depth, state: GameState, alpha, beta):
        #chattyish
        v = float('inf')
        actions = state.getLegalActions(agentIndex)
        if not actions:
            return self.evaluationFunction(state)
        
        nextAgent = agentIndex + 1
        if nextAgent >= state.getNumAgents():
            #pacman's turn if this was the last ghost
            nextAgent = 0

        for action in actions:
            successorState = state.generateSuccessor(agentIndex, action)
            if nextAgent == 0:  
                nextAgentScore = self.alphaBeta(successorState, 0, depth + 1, alpha, beta)
                #finish pacman and ghost turn so depth increases
                v = min(v, nextAgentScore)
            else: 
                nextAgentScore = self.alphaBeta(successorState, nextAgent, depth, alpha, beta)
                #next ghost's turn
                v = min(v, nextAgentScore)
            
            #prune
            if v < alpha:
                return v
            beta = min(beta, v)
        return v
    

    

class ExpectimaxAgent(MultiAgentSearchAgent):
    #FULLY CHATTED
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
        bestAction = None
        maxScore = float('-inf')

        # Pacman is always agentIndex = 0, trying to maximize its score
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = self.expectimax(1, 0, successor)
            if score > maxScore:
                maxScore = score
                bestAction = action  # Keep track of the best action
        
        return bestAction

    def expectimax(self, agentIndex, depth, state):
        """
        Recursively performs expectimax search. Returns the value of the state.
        """
        # Base case: If we've reached the depth limit or a terminal state
        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        # Pacman's turn (maximizing agent)
        if agentIndex == 0:
            return self.pacmanMaxValue(state, depth)
        
        # Ghosts' turn (chance nodes with uniform probability)
        else:
            return self.ghostExpectedValue(agentIndex, depth, state)

    def pacmanMaxValue(self, state, depth):
        """
        Pacman is the maximizer, trying to choose the best action for the highest score
        """
        v = float('-inf')  # Initialize to the worst possible value for maximizer
        actions = state.getLegalActions(0)  # Get Pacman's legal moves
        
        if not actions:
            return self.evaluationFunction(state)

        for action in actions:
            successor = state.generateSuccessor(0, action)
            v = max(v, self.expectimax(1, depth, successor))  # Pacman maximizes the score
        return v

    def ghostExpectedValue(self, agentIndex, depth, state):
        """
        Ghosts are modeled as chance nodes. Each ghost moves uniformly at random.
        We calculate the expected value of all possible ghost actions.
        """
        actions = state.getLegalActions(agentIndex)
        if not actions:
            return self.evaluationFunction(state)

        # Calculate the expected value of the ghost's actions
        totalValue = 0
        prob = 1 / len(actions)  # Uniform probability for each action

        nextAgent = agentIndex + 1
        if nextAgent >= state.getNumAgents():  # After the last ghost, go back to Pacman
            nextAgent = 0

        for action in actions:
            successor = state.generateSuccessor(agentIndex, action)
            if nextAgent == 0:  # If Pacman's turn is next, increase depth
                totalValue += prob * self.expectimax(0, depth + 1, successor)
            else:  # If it's another ghost's turn, stay at the same depth
                totalValue += prob * self.expectimax(nextAgent, depth, successor)

        return totalValue
    

def betterEvaluationFunction(currentGameState: GameState):
    #FULLY CHATTED
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>

    Explanation of the Components:
    Distance to Food:

    The function penalizes Pacman for being far from food. The closer Pacman is to food, the higher the score. This encourages Pacman to prioritize getting food quickly.
    1.0 / nearestFoodDist is added to the score, so if Pacman is close to food, this term becomes larger.
    Distance to Ghosts:

    If ghosts are active (not scared), Pacman is penalized for being close to them. The closer Pacman is to an active ghost, the more the score is reduced.
    If ghosts are scared (due to a power pellet), Pacman is rewarded for being close, because it can eat them for extra points.
    Number of Food Pellets Remaining:

    Pacman is rewarded as the number of food pellets left decreases, which incentivizes Pacman to clear the board quickly.
    Distance to Power Pellets (Capsules):

    Pacman is rewarded for being near power pellets, encouraging it to eat the power pellets to turn ghosts into edible targets.
    Endgame States:

    If Pacman reaches a winning state, the score is set to infinity to represent the best possible outcome.
    If Pacman loses, the score is set to -infinity to represent the worst possible outcome.
    Tuning and Testing:
    You can adjust the weights of the different components (e.g., the multiplier for ghost distance or food distance) based on how Pacman performs during testing. You can run the autograder and analyze the results to see how well your agent performs, and tweak the evaluation function if needed.

    How This Evaluation Function Works:
    It balances different objectives: Pacman is incentivized to go after food, avoid dangerous ghosts, chase scared ghosts, and consume power pellets.
    By combining these factors into a single score, Pacman can make more informed decisions about which game states are preferable.



    """
    "*** YOUR CODE HERE ***"
    # Get useful information from the game state
    pacmanPos = currentGameState.getPacmanPosition()
    foodGrid = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    capsulePositions = currentGameState.getCapsules()
    
    # Initialize score with the current game state's score
    score = currentGameState.getScore()
    
    # 1. Add the negative distance to the nearest food
    foodList = foodGrid.asList()  # Convert food grid to a list of food positions
    if foodList:
        nearestFoodDist = min([manhattanDistance(pacmanPos, food) for food in foodList])
        score += 1.0 / nearestFoodDist  # The closer Pacman is to food, the higher the score
    
    # 2. Add penalty for being too close to active ghosts, and bonus for being close to scared ghosts
    for ghost in ghostStates:
        ghostPos = ghost.getPosition()
        ghostDist = manhattanDistance(pacmanPos, ghostPos)
        if ghost.scaredTimer > 0:
            # Ghost is scared, prioritize chasing it
            score += 200 / (ghostDist + 1)  # Add bonus for being close to scared ghosts
        else:
            # Ghost is not scared, avoid it
            if ghostDist > 0:
                score -= 10 / ghostDist  # Subtract points for being close to active ghosts
    
    # 3. Add bonus for having fewer food pellets left
    score += 10 * (currentGameState.getNumFood() - len(foodList))  # Fewer food left increases score

    # 4. Add bonus for being close to power pellets (capsules)
    if capsulePositions:
        nearestCapsuleDist = min([manhattanDistance(pacmanPos, capsule) for capsule in capsulePositions])
        score += 5.0 / nearestCapsuleDist  # Add small bonus for being close to power pellets

    # 5. Consider the win and lose states (endgame situations)
    if currentGameState.isWin():
        return float('inf')  # Winning is the best state, so return maximum possible value
    if currentGameState.isLose():
        return float('-inf')  # Losing is the worst state, so return minimum possible value

    return score

# Abbreviation
better = betterEvaluationFunction
