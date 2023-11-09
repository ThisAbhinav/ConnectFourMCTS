import math


# sets the game meta data
class GameMeta:
    PLAYERS = {"none": 0, "one": 1, "two": 2}
    OUTCOMES = {"none": 0, "one": 1, "two": 2, "draw": 3}
    INF = float("inf")
    ROWS = 6
    COLS = 7


# sets the MCTS meta data
class MCTSMeta:
    EXPLORATION = math.sqrt(2)  # exploration rate of the agent
