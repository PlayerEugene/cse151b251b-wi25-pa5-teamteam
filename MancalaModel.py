import torch
import torch.nn as nn
from engine import MancalaGame
import random
import math
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import copy

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

class MancalaModelMCTSPolicy:
    def __init__(self, 
                 model: MancalaModel,
                 c_puct=1.4,
                 n_simulations=50,
                 dirichlet_alpha=0.03,
                 epsilon=0.25):

        self.model = model
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.epsilon = epsilon

        # MCTS statistics:
        self.N = {}
        self.W = {}
        self.Q = {}
        self.P = {}

        # Flag for whether to add noise to the root node (self-play).
        self.add_dirichlet_noise = False

    def _reset_mcts(self):
        """
        Clear MCTS data so each new game has its own fresh search tree.
        """
        self.N.clear()
        self.W.clear()
        self.Q.clear()
        self.P.clear()

    def _get_state_key(self, game: MancalaGame) -> str:
        board_str = ",".join(map(str, game.board))
        return f"{board_str}-{game.current_player}"

    def _predict(self, game: MancalaGame):
        """
        Forward pass through the neural network to get policy (move_probs) and value.
        """
        in_tensor = torch.tensor(
            game.board[0:6] + game.board[7:13] + [game.current_player],
            dtype=torch.float32
        ).unsqueeze(0)
        with torch.no_grad():
            move_probs, value = self.model(in_tensor)
        move_probs = move_probs[0].cpu().numpy()
        value = value[0].item()
        return move_probs, value

    def _expand_node(self, game: MancalaGame, state_key: str):
        policy, value = self._predict(game)
        valid_moves = game.get_valid_moves()

        # Adjust probabilities for valid moves only:
        for idx in range(len(policy)):
            pocket = idx if idx < 6 else idx + 1
            if pocket not in valid_moves:
                policy[idx] = 0.0

        sum_p = sum(policy)
        if sum_p > 1e-8:
            policy = policy / sum_p
        else:
            # If all probabilities are zero, distribute uniformly among valid moves.
            for idx in range(len(policy)):
                pocket = idx if idx < 6 else idx + 1
                if pocket in valid_moves:
                    policy[idx] = 1.0
            policy /= sum(policy)

        # Add Dirichlet noise if we are at the root node (self-play training).
        if self.add_dirichlet_noise:
            dirichlet_input = [
                self.dirichlet_alpha if (idx if idx < 6 else idx + 1) in valid_moves else 0.0001 
                for idx in range(len(policy))
            ]
            noise = np.random.dirichlet(dirichlet_input)
            for i in range(len(policy)):
                policy[i] = (1 - self.epsilon) * policy[i] + self.epsilon * noise[i]
            policy_sum = sum(policy)
            if policy_sum > 1e-8:
                policy /= policy_sum

        # Store MCTS policy, initialize stats
        self.P[state_key] = policy
        self.N[state_key] = {}
        self.W[state_key] = {}
        self.Q[state_key] = {}
        for mv in valid_moves:
            self.N[state_key][mv] = 0
            self.W[state_key][mv] = 0
            self.Q[state_key][mv] = 0

        return value

    def _ucb_score(self, s_key: str, a: int, parent_sum_visits: int):
        a_idx = a if a < 6 else a - 1
        q_val = self.Q[s_key][a]
        p_val = self.P[s_key][a_idx]
        n_val = self.N[s_key][a]
        return q_val + self.c_puct * p_val * math.sqrt(parent_sum_visits) / (1 + n_val)

    def _simulate(self, game: MancalaGame):
        """
        One MCTS simulation from the current state to expand or reach terminal.
        """
        from copy import deepcopy
        state_history = []
        current_game = deepcopy(game)
        state_key = self._get_state_key(current_game)

        while True:
            valid_moves = current_game.get_valid_moves()
            if current_game.is_game_over() or state_key not in self.P:
                break
            parent_visits = sum(self.N[state_key].values())
            best_action = None
            best_ucb = -float('inf')
            for a in valid_moves:
                ucb = self._ucb_score(state_key, a, parent_visits)
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_action = a
            state_history.append((state_key, best_action))
            current_game.make_move(best_action)
            state_key = self._get_state_key(current_game)
            if current_game.is_game_over():
                break

        # Leaf node:
        if not current_game.is_game_over():
            leaf_value = self._expand_node(current_game, state_key)
        else:
            # Terminal node, compute value from winner:
            p1_score, p2_score = current_game.get_score()
            if p1_score == p2_score:
                leaf_value = 0.0
            else:
                root_player = game.current_player
                winner = 1 if p1_score > p2_score else 2
                leaf_value = 1.0 if winner == root_player else -1.0

        # Backpropagate:
        cur_value = leaf_value
        for (prev_key, action_taken) in reversed(state_history):
            self.N[prev_key][action_taken] += 1
            self.W[prev_key][action_taken] += cur_value
            self.Q[prev_key][action_taken] = self.W[prev_key][action_taken] / self.N[prev_key][action_taken]
            cur_value = -cur_value

    def get_action_prob(self, game: MancalaGame, temp=1.0, add_dirichlet_noise=False):
        """
        Return the MCTS-based policy (a probability distribution over all 12 pockets).
        """
        self.add_dirichlet_noise = add_dirichlet_noise
        # Run MCTS simulations from the current game state
        for _ in range(self.n_simulations):
            self._simulate(game)
        state_key = self._get_state_key(game)
        valid_moves = game.get_valid_moves()

        counts = [0] * 12
        if state_key not in self.N:
            # If we somehow never expanded, fallback to uniform among valid moves
            for idx in range(12):
                pocket = idx if idx < 6 else idx + 1
                if pocket in valid_moves:
                    counts[idx] = 1
        else:
            # Use visit counts
            for mv in valid_moves:
                idx = mv if mv < 6 else mv - 1
                counts[idx] = self.N[state_key][mv]

        # Softmax or argmax over counts, based on temp
        if temp < 1e-8:
            # Argmax
            best_idx = max(range(12), key=lambda i: counts[i])
            policy = [0] * 12
            policy[best_idx] = 1.0
            return policy
        else:
            counts_exp = [c**(1.0 / temp) for c in counts]
            total = sum(counts_exp)
            if total < 1e-8:
                # If all counts are 0, distribute uniformly among valid moves.
                policy = [0] * 12
                for idx in range(12):
                    pocket = idx if idx < 6 else idx + 1
                    if pocket in valid_moves:
                        policy[idx] = 1
                sm = sum(policy)
                policy = [p / sm for p in policy]
            else:
                policy = [x / total for x in counts_exp]
            return policy

    def play_self_game(self, temp=1.0):
        """
        Plays one full game in self-play mode. 
        Returns a list of (state, π, z) for training.
        """
        train_examples = []
        from copy import deepcopy

        game = MancalaGame()
        self._reset_mcts()  # Start a fresh search tree for this game.
        history = []
        
        while not game.is_game_over():
            policy = self.get_action_prob(game, temp=temp, add_dirichlet_noise=True)
            state_vec = game.board[0:6] + game.board[7:13] + [game.current_player]
            history.append((state_vec, policy, game.current_player))

            action_idx = random.choices(range(12), weights=policy, k=1)[0]
            pocket = action_idx if action_idx < 6 else action_idx + 1
            game.make_move(pocket)

        p1_score, p2_score = game.get_score()
        if p1_score > p2_score:
            winner = 1
        elif p2_score > p1_score:
            winner = 2
        else:
            winner = 0

        # Convert all steps to final training data:
        for (state_vec, pi, cur_player) in history:
            if winner == 0:
                z = 0
            else:
                z = 1 if (winner == cur_player) else -1
            train_examples.append((state_vec, pi, z))

        return train_examples

    def _train_on_examples(self, examples, batch_size=64, epochs=1, lr=1e-3):
        """
        Train the network on the provided examples (state, pi, z).
        This is extracted from your `train_self_play(...)` code but modularized.
        """
        states = []
        pis = []
        zs = []
        for (state_vec, pi, z) in examples:
            states.append(state_vec)
            pis.append(pi)
            zs.append(z)

        states_t = torch.tensor(states, dtype=torch.float32)
        pis_t = torch.tensor(pis, dtype=torch.float32)
        zs_t = torch.tensor(zs, dtype=torch.float32).unsqueeze(-1)

        dataset = TensorDataset(states_t, pis_t, zs_t)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        xent = nn.CrossEntropyLoss()
        mse = nn.MSELoss()

        self.model.train()
        for _ in range(epochs):
            for batch_s, batch_pi, batch_z in dataloader:
                optimizer.zero_grad()
                out_pi, out_v = self.model(batch_s)
                
                # Policy loss (cross-entropy but with softmax input directly)
                log_probs = torch.log(out_pi + 1e-7)
                policy_loss = -(batch_pi * log_probs).sum(dim=1).mean()
                
                # Value loss (MSE)
                value_loss = mse(out_v, batch_z)
                
                loss = policy_loss + value_loss
                loss.backward()
                optimizer.step()

        self.model.eval()

    def pit(self, opponent_policy, n_games=50):
        """
        Pit this policy (self) against 'opponent_policy' for n_games.
        Return fraction of games that 'self' wins.
        Each side can go first half the time (optional).
        """
        self_wins = 0
        for game_idx in range(n_games):
            # Alternate who goes first, for fairness
            # (Assuming MancalaGame can set current_player manually, or
            #  just rely on the default. You might vary who starts if desired.)
            
            game = MancalaGame()
            self._reset_mcts()
            opponent_policy._reset_mcts()
            
            if game_idx % 2 == 1:
                game.current_player = 2

            while not game.is_game_over():
                if game.current_player == 1:
                    # use 'self' to pick a move
                    policy = self.get_action_prob(game, temp=0.2, add_dirichlet_noise=False)
                    action_idx = random.choices(range(len(policy)), weights=policy, k=1)[0]
                else:
                    # use the opponent policy
                    policy = opponent_policy.get_action_prob(game, temp=0.2, add_dirichlet_noise=False)
                    action_idx = random.choices(range(len(policy)), weights=policy, k=1)[0]

                pocket = action_idx if action_idx < 6 else action_idx + 1
                game.make_move(pocket)

            p1_score, p2_score = game.get_score()
            winner = 1 if (p1_score > p2_score) else (2 if p2_score > p1_score else 0)
            if winner == 1:
                self_wins += 1
        return self_wins / n_games

    def train_policy_iteration(self, 
                               num_iters=10, 
                               n_games_per_iter=10, 
                               pit_games=10, 
                               threshold=0.55,
                               batch_size=64, 
                               epochs=1, 
                               lr=1e-3):
        """
        Full AlphaZero-style training loop:
          - For num_iters:
            1) Collect training data from self-play with the current best model
            2) Train a candidate ("new") model on that data
            3) Pit the new model vs. best model
            4) If new model wins above threshold, adopt it
        """
        best_model = copy.deepcopy(self.model)

        for iteration in range(num_iters):
            print(f"\n--- Iteration {iteration+1}/{num_iters} ---")
            iteration_examples = []
            
            # Collect self-play data using the current best model
            # (Use a separate MCTSPolicy object that wraps best_model)
            best_policy = MancalaModelMCTSPolicy(
                best_model, 
                c_puct=self.c_puct,
                n_simulations=self.n_simulations,
                dirichlet_alpha=self.dirichlet_alpha,
                epsilon=self.epsilon
            )

            for _ in range(n_games_per_iter):
                iteration_examples.extend(best_policy.play_self_game(temp=1.0))

            # Create a new model starting from the best model's parameters
            new_model = copy.deepcopy(best_model)
            new_policy = MancalaModelMCTSPolicy(
                new_model, 
                c_puct=self.c_puct,
                n_simulations=self.n_simulations,
                dirichlet_alpha=self.dirichlet_alpha,
                epsilon=self.epsilon
            )

            # Train new_model on the self-play examples
            new_policy._train_on_examples(
                iteration_examples, 
                batch_size=batch_size, 
                epochs=epochs, 
                lr=lr
            )

            # Now pit new_policy vs best_policy
            win_rate = new_policy.pit(best_policy, n_games=pit_games)
            print(f"New model win rate = {win_rate*100:.2f}%")

            if win_rate >= threshold:
                print("New model surpasses threshold -> Accepting new model.")
                best_model = copy.deepcopy(new_model)
            else:
                print("New model did not surpass threshold -> Keeping old model.")

        self.model.load_state_dict(best_model.state_dict())
        print("\nTraining finished. Best model is now loaded into self.model.")
    
    def save_policy_weights(policy, filepath):
        """
        Save the neural network weights of the MCTS policy to a file.
        
        Args:
            policy: The MancalaModelMCTSPolicy instance
            filepath: Path where to save the weights
        """
        torch.save(policy.model.state_dict(), filepath)
    
    def load_policy_weights(policy, filepath):
        """
        Load neural network weights into the MCTS policy from a file.
        
        Args:
            policy: The MancalaModelMCTSPolicy instance
            filepath: Path from where to load the weights
        """
        policy.model.load_state_dict(torch.load(filepath))
        policy.model.eval()  # Set to evaluation mode