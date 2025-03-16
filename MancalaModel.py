import torch
import torch.nn as nn
from engine import MancalaGame
import random
import math
import torch
import torch.optim as optim

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
    def __init__(self, num_simulations=50, ucb_c=1.4, mcts_player=1):
        """
        :param num_simulations: How many playouts to run per move
        :param ucb_c: The exploration constant for UCB1
        :param mcts_player: Which player this MCTS is playing for (1 or 2)
        """
        self.num_simulations = num_simulations
        self.ucb_c = ucb_c
        self.mcts_player = mcts_player

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
            best_score = float("-inf")
            best_children = []
            for child in self.children:
                if child.visit_count == 0:
                    ucb = float("inf")
                else:
                    avg_value = child.total_value / child.visit_count
                    ucb = avg_value + c * math.sqrt(
                        math.log(self.visit_count) / child.visit_count
                    )
                if ucb > best_score:
                    best_score = ucb
                    best_children = [child]
                elif abs(ucb - best_score) < 1e-9:
                    best_children.append(child)
            return random.choice(best_children)

    def mcts(self, root_game_state: MancalaGame):
        root_node = self.Node(game_state=copy_game_state(root_game_state),
                              parent=None, move=None)

        for _ in range(self.num_simulations):
            selected_node = self._selection(root_node)
            expanded_node = self._expansion(selected_node)
            rollout_result = self._simulation(expanded_node.game_state)
            self._backpropagation(expanded_node, rollout_result)

        best_child = self._choose_best_move(root_node)
        return best_child.move

    def _selection(self, node: Node) -> Node:
        """Select until we hit a node that can still expand or is terminal."""
        while not node.is_terminal_node() and node.is_fully_expanded():
            node = node.best_child(c=self.ucb_c)
        return node

    def _expansion(self, node: Node) -> Node:
        """Expand one child if possible."""
        if node.is_terminal_node():
            return node
        if node.untried_moves:
            move = random.choice(node.untried_moves)
            node.untried_moves.remove(move)
            new_game_state = copy_game_state(node.game_state)
            new_game_state.make_move(move)
            child_node = self.Node(game_state=new_game_state, parent=node, move=move)
            node.children.append(child_node)
            return child_node
        return node

    def _simulation(self, temp_game_state: MancalaGame):
        """Roll out (simulate) until game is over, returning winner (1,2) or 0 for tie."""
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
        """
        result = 1 means Player 1 won, 2 means Player 2 won, 0 means tie.
        But we want to assign +1 reward if `mcts_player` is the winner,
        0.5 for tie, and 0 for a loss.
        """
        if result == 0:
            reward = 0.5
        elif result == self.mcts_player:
            reward = 1.0
        else:
            reward = 0.0

        while node is not None:
            node.visit_count += 1
            node.total_value += reward
            node = node.parent

    def _choose_best_move(self, root_node: Node) -> Node:
        """Choose the child with the highest visit count."""
        best_visit_count = -1
        best_nodes = []
        for child in root_node.children:
            if child.visit_count > best_visit_count:
                best_visit_count = child.visit_count
                best_nodes = [child]
            elif child.visit_count == best_visit_count:
                best_nodes.append(child)
        return random.choice(best_nodes)

class SimpleMCTSPolicy:
    def __init__(self, model, c_puct=1.4, n_simulations=50):
        self.model = model
        self.mcts_agent = MancalaModelMCTS(num_simulations=n_simulations, ucb_c=c_puct)

    def _run_mcts(self, game):
        root = self.mcts_agent.Node(game_state=copy_game_state(game))
        for _ in range(self.mcts_agent.num_simulations):
            sel = self.mcts_agent._selection(root)
            exp = self.mcts_agent._expansion(sel)
            res = self.mcts_agent._simulation(exp.game_state)
            self.mcts_agent._backpropagation(exp, res)
        return root

    def get_action_prob(self, game):
        r = self._run_mcts(game)
        d = [0]*12
        s = 0
        for c in r.children:
            idx = c.move if c.move < 6 else c.move - 1
            d[idx] = c.visit_count
            s += c.visit_count
        if s < 1e-8: s = 1
        return [x/s for x in d]

    def inference_move(self, game):
        p = self.get_action_prob(game)
        i = max(range(len(p)), key=lambda k: p[k])
        return i if i < 6 else i + 1

    def train_self_play(self, n_games=10, batch_size=64, epochs=1, lr=1e-3):
        data = []
        for _ in range(n_games):
            g = MancalaGame()
            hist = []
            while not g.is_game_over():
                ap = self.get_action_prob(g)
                st = g.board[0:6] + g.board[7:13] + [g.current_player]
                mv = random.choices(range(12), weights=ap, k=1)[0]
                hist.append((st, ap, g.get_current_player()))
                g.make_move(mv if mv < 6 else mv + 1)
            p1, p2 = g.get_score()
            w = 1 if p1 > p2 else (2 if p2 > p1 else 0)
            for s, p, cp in hist:
                z = 0 if w == 0 else (1 if w == cp else -1)
                data.append((s, p, z))
        X = torch.tensor([d[0] for d in data], dtype=torch.float32)
        Py = torch.tensor([d[1] for d in data], dtype=torch.float32)
        Zy = torch.tensor([d[2] for d in data], dtype=torch.float32).unsqueeze(-1)
        ds = torch.utils.data.TensorDataset(X, Py, Zy)
        dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
        opt = optim.Adam(self.model.parameters(), lr=lr)
        ce = nn.CrossEntropyLoss()
        mse = nn.MSELoss()
        self.model.train()
        for _ in range(epochs):
            for xb, pb, zb in dl:
                opt.zero_grad()
                op, ov = self.model(xb)
                l1 = -(pb * torch.log(op+1e-8)).sum(dim=1).mean()
                l2 = mse(ov, zb)
                (l1+l2).backward()
                opt.step()
        self.model.eval()