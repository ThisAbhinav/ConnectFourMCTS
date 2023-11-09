import random
import time
import math
from copy import deepcopy

from ConnectState import ConnectState
from meta import GameMeta, MCTSMeta


class Node:
    def __init__(self, move, parent):
        # exploitation term
        self.Q = 0  # number of wins from that node
        self.N = 0  # total simulations done from that node

        # exploration term
        self.parent = parent  # parent of the given node
        self.move = move  # move that led to the node
        self.children = {}
        self.outcome = GameMeta.PLAYERS["none"]

    def add_children(self, children: dict) -> None:  # add children to the node
        for child in children:
            self.children[child.move] = child

    def value(self, explore: float = MCTSMeta.EXPLORATION):  # UCT value calculation
        if self.N == 0:
            return 0 if explore == 0 else GameMeta.INF
        else:
            return self.Q / self.N + explore * math.sqrt(
                math.log(self.parent.N) / self.N
            )


class MCTS:
    def __init__(self, state=ConnectState()):
        self.root_state = deepcopy(state)
        self.root = Node(None, None)
        self.run_time = 0
        self.node_count = 0
        self.num_rollouts = 0

    def select_node(self) -> tuple:
        node = self.root
        state = deepcopy(self.root_state)

        while len(node.children) != 0:
            # Goes through all the children of the node and selects the one with
            # the highest UCT value and expands upon it till it reaches a terminal state
            children = node.children.values()  # get all children of the node
            max_value = max(children, key=lambda n: n.value()).value()

            # select nodes with the highest UCT value
            max_nodes = [n for n in children if n.value() == max_value]

            # randomly select which max node to expand upon
            node = random.choice(max_nodes)
            # node.move gives the move that led to the node from parent. as we are deciding the best move from parent. we need to move the state to the node(child of parent)
            state.move(node.move)

            # if the node is a first time visit, return the node and the state
            if node.N == 0:
                return node, state

        # landed on a node with no children but visited before
        if self.expand(node, state):
            # expand the just 1 level deeper when we don't have knowledge.
            node = random.choice(list(node.children.values()))
            # randomly select a child of the node
            state.move(node.move)  # move the state to the random chosen child node

        return node, state

    # expand the node by adding all the children of the node
    def expand(self, parent: Node, state: ConnectState) -> bool:
        if state.game_over():
            return False

        children = [Node(move, parent) for move in state.get_legal_moves()]
        parent.add_children(children)

        return True

    # randomly play the game till the end(tie or win)
    def roll_out(self, state: ConnectState) -> int:
        while not state.game_over():
            state.move(random.choice(state.get_legal_moves()))

        return state.get_outcome()

    # back propagate the result, when the game is over
    def back_propagate(self, node: Node, turn: int, outcome: int) -> None:
        reward = 0 if outcome == turn else 1

        while node is not None:
            node.N += 1
            node.Q += reward
            node = node.parent
            if outcome == GameMeta.OUTCOMES["draw"]:
                reward = 0  # draw
            else:
                reward = 1 - reward  # switch reward

    def search(self, time_limit: int):
        start_time = time.process_time()

        num_rollouts = 0
        while time.process_time() - start_time < time_limit:
            node, state = self.select_node()  # select a node to expand upon
            outcome = self.roll_out(state)  # randomly play the game till the end
            self.back_propagate(
                node, state.to_play, outcome
            )  # back propagate the result
            num_rollouts += 1  # increment the number of rollouts

        run_time = time.process_time() - start_time
        self.run_time = run_time
        self.num_rollouts = num_rollouts

    def best_move(self):
        if self.root_state.game_over():
            return -1
        # select the child with the highest number of visits
        max_value = max(self.root.children.values(), key=lambda n: n.N).N
        max_nodes = [n for n in self.root.children.values() if n.N == max_value]
        best_child = random.choice(max_nodes)
        return best_child.move

    def move(self, move):
        if move in self.root.children:
            self.root_state.move(move)
            self.root = self.root.children[move]
            return

        self.root_state.move(move)
        self.root = Node(None, None)

    def statistics(self) -> tuple:
        return self.num_rollouts, self.run_time
