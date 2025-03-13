import torch
import torch.nn as nn
from engine import MancalaGame
import random
import math

def copy_game_state(game: MancalaGame) -> MancalaGame:
    new_game = MancalaGame()
    new_game.board = game.board.copy()
    new_game.current_player = game.current_player
    return new_game

class MancalaModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)

        self.lin1 = nn.Linear(13, 128)
        self.lin2 = nn.Linear(128, 128)
        
        self.policy_head = nn.Linear(128, 12)  
        self.value_head = nn.Linear(128, 1) 

    def forward(self, x):
        x1 = self.relu(self.lin1(x))
        x2 = self.relu(self.lin2(x1))

        move_probs = self.policy_head(x2)
        move_probs = torch.softmax(move_probs, dim=-1)

        state_value = self.value_head(x2)
        state_value = torch.tanh(state_value)

        return move_probs, state_value

class MancalaModelMCTS:
    def __init__(self, num_simulations=1000, ucb_c=1.4):
        self.num_simulations = num_simulations
        self.ucb_c = ucb_c

    class Node:
        def __init__(self, game_state: MancalaGame, parent=None, move=None):
            self.game_state = game_state

            self.move = move

            self.parent = parent

            self.children = []

            self.untried_moves = game_state.get_valid_moves()

            self.visit_count = 0

            self.total_value = 0.0

        def is_fully_expanded(self):
            return len(self.untried_moves) == 0

        def is_terminal_node(self):
            return self.game_state.is_game_over()

        def best_child(self, c=1.4):
            best_score = -float('inf')
            best_children = []

            for child in self.children:
                if child.visit_count == 0:
                    ucb = float('inf')
                else:
                    avg_value = child.total_value / child.visit_count
                    ucb = avg_value + c * math.sqrt(
                        (math.log(self.visit_count)) / child.visit_count
                    )
                if ucb > best_score:
                    best_score = ucb
                    best_children = [child]
                elif abs(ucb - best_score) < 1e-9:
                    best_children.append(child)

            return random.choice(best_children)

    def mcts(self, root_game_state: MancalaGame):
        root_node = self.Node(game_state=copy_game_state(root_game_state), parent=None, move=None)

        for _ in range(self.num_simulations):
            selected_node = self._selection(root_node)

            expanded_node = self._expansion(selected_node)

            rollout_result = self._simulation(expanded_node.game_state)

            self._backpropagation(expanded_node, rollout_result)

        best_child = self._choose_best_move(root_node)
        return best_child.move

    def _selection(self, node: Node) -> Node:
        while not node.is_terminal_node() and node.is_fully_expanded():
            node = node.best_child(c=self.ucb_c)
        return node

    def _expansion(self, node: Node) -> Node:
        if node.is_terminal_node():
            return node

        if node.untried_moves:
            move = random.choice(node.untried_moves)
            node.untried_moves.remove(move)

            new_game_state = copy_game_state(node.game_state)
            valid = new_game_state.make_move(move)

            child_node = self.Node(game_state=new_game_state, parent=node, move=move)
            node.children.append(child_node)
            return child_node

        return node

    def _simulation(self, temp_game_state: MancalaGame):
        sim_game = copy_game_state(temp_game_state)

        while not sim_game.is_game_over():
            moves = sim_game.get_valid_moves()
            if not moves:
                break
            move = random.choice(moves)
            sim_game.make_move(move)

        p1_score, p2_score = sim_game.get_score()
        if p1_score > p2_score:
            return 1
        elif p2_score > p1_score:
            return 2
        else:
            return 0

    def _backpropagation(self, node: Node, result: int):
        reward_map = {
            1: 1.0,   # P1 wins
            2: 0.0,   # P2 wins
            0: 0.5    # tie
        }
        reward = reward_map[result]

        while node is not None:
            node.visit_count += 1

            node.total_value += reward

            node = node.parent

    def _choose_best_move(self, root_node: Node) -> Node:
        best_visit_count = -1
        best_children = []
        for child in root_node.children:
            if child.visit_count > best_visit_count:
                best_visit_count = child.visit_count
                best_children = [child]
            elif child.visit_count == best_visit_count:
                best_children.append(child)

        return random.choice(best_children)
