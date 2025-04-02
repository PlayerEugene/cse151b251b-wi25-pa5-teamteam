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

class BaseMancalaModel(nn.Module):
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

class MancalaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(13, 128)
        self.ln1 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 64)
        self.ln2 = nn.LayerNorm(64)
        self.dropout = nn.Dropout(p=0.1)
        self.relu = nn.ReLU(inplace=True)
        
        self.policy_head = nn.Linear(64, 12)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        x = self.relu(self.ln1(self.fc1(x)))
        x = self.relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)

        move_probs = self.policy_head(x)
        move_probs = torch.softmax(move_probs, dim=-1)

        state_value = self.value_head(x)
        state_value = torch.tanh(state_value)

        return move_probs, state_value

class MancalaModelv2(nn.Module):
    def __init__(self):
        super().__init__()
        # Increase initial layer size for better feature extraction
        self.fc1 = nn.Linear(13, 256)  # Increased from 128
        self.ln1 = nn.LayerNorm(256)
        
        # Add more layers for deeper pattern recognition
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)
        
        self.fc3 = nn.Linear(128, 64)  # New layer
        self.ln3 = nn.LayerNorm(64)
        
        # Slightly increase dropout for better generalization
        self.dropout = nn.Dropout(p=0.15)  # Increased from 0.1
        self.relu = nn.LeakyReLU(inplace=True)  # Changed from ReLU to LeakyReLU
        
        # Heads remain the same size but get separate dropouts
        self.policy_dropout = nn.Dropout(p=0.1)
        self.value_dropout = nn.Dropout(p=0.1)
        self.policy_head = nn.Linear(64, 12)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        x = self.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        
        x = self.relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        
        x = self.relu(self.ln3(self.fc3(x)))
        x = self.dropout(x)

        # Separate paths for policy and value
        policy = self.policy_dropout(x)
        value = self.value_dropout(x)

        move_probs = self.policy_head(policy)
        move_probs = torch.softmax(move_probs, dim=-1)

        state_value = self.value_head(value)
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
                 model: MancalaModelv2,
                 c_puct=1.4,
                 n_simulations=50,
                 dirichlet_alpha=0.03,
                 epsilon=0.25,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):

        self.model = model.to(device)
        self.device = device
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
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)
        
        with torch.no_grad():
            move_probs, value = self.model(in_tensor)
        
        move_probs = move_probs[0].cpu().numpy()
        value = value[0].cpu().item()
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
        """
        states = []
        pis = []
        zs = []
        for (state_vec, pi, z) in examples:
            states.append(state_vec)
            pis.append(pi)
            zs.append(z)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        pis_t = torch.tensor(pis, dtype=torch.float32, device=self.device)
        zs_t = torch.tensor(zs, dtype=torch.float32, device=self.device).unsqueeze(-1)

        dataset = TensorDataset(states_t, pis_t, zs_t)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
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

    def pit(self, opponent_policy, n_games=50, opponent_is_mcts=False, mcts_simulations=50):
        """
        Pit this policy against either another policy or pure MCTS.
        
        Args:
            opponent_policy: Either another MCTSPolicy instance or an MCTS instance
            n_games: Number of games to play
            opponent_is_mcts: If True, opponent_policy is treated as pure MCTS
            mcts_simulations: Number of MCTS simulations if opponent_is_mcts=True
        """
        self_wins = 0
        for game_idx in range(n_games):
            game = MancalaGame()
            self._reset_mcts()
            if not opponent_is_mcts:
                opponent_policy._reset_mcts()
            else:
                # Create fresh MCTS opponent for each game with correct parameters
                opponent_policy = MancalaModelMCTS(
                    num_simulations=mcts_simulations,
                    ucb_c=1.4,
                    mcts_player=2 if game_idx % 2 == 0 else 1
                )
            
            # Alternate who plays as Player 1
            we_play_as_p1 = (game_idx % 2 == 0)
            if not we_play_as_p1:
                game.current_player = 2

            while not game.is_game_over():
                is_our_turn = (game.current_player == 1) == we_play_as_p1
                
                if is_our_turn:
                    # use 'self' to pick a move
                    policy = self.get_action_prob(game, temp=0.2, add_dirichlet_noise=False)
                    action_idx = random.choices(range(len(policy)), weights=policy, k=1)[0]
                    pocket = action_idx if action_idx < 6 else action_idx + 1
                else:
                    # use the opponent (either MCTS or policy)
                    if opponent_is_mcts:
                        pocket = opponent_policy.mcts(game)
                    else:
                        policy = opponent_policy.get_action_prob(game, temp=0.2, add_dirichlet_noise=False)
                        action_idx = random.choices(range(len(policy)), weights=policy, k=1)[0]
                        pocket = action_idx if action_idx < 6 else action_idx + 1
                
                game.make_move(pocket)

            p1_score, p2_score = game.get_score()
            winner = 1 if (p1_score > p2_score) else (2 if p2_score > p1_score else 0)
            if (winner == 1 and we_play_as_p1) or (winner == 2 and not we_play_as_p1):
                self_wins += 1
        return self_wins / n_games

    def train_policy_iteration(self, 
                               num_iters=10, 
                               n_games_per_iter=10, 
                               pit_games=10, 
                               threshold=0.55,
                               batch_size=64, 
                               epochs=1, 
                               lr=1e-3,
                               base_model_path=None):
        if base_model_path:
            print(f"Loading base model from {base_model_path}")
            self.model.load_state_dict(torch.load(base_model_path))
            best_model = copy.deepcopy(self.model)
        else:
            best_model = copy.deepcopy(self.model)

        # Create pure MCTS opponent
        mcts_opponent = MancalaModelMCTS(num_simulations=50, mcts_player=2)
        
        # Track best model's MCTS performance
        best_policy = MancalaModelMCTSPolicy(
            best_model, 
            c_puct=self.c_puct,
            n_simulations=self.n_simulations,
            dirichlet_alpha=self.dirichlet_alpha,
            epsilon=self.epsilon
        )
        best_mcts_win_rate = best_policy.pit(mcts_opponent, n_games=pit_games, opponent_is_mcts=True)
        print(f"Initial model vs MCTS win rate = {best_mcts_win_rate*100:.2f}%")

        for iteration in range(num_iters):
            print(f"\n--- Iteration {iteration+1}/{num_iters} ---")
            iteration_examples = []
            
            # Collect self-play data using the current best model
            best_policy = MancalaModelMCTSPolicy(
                best_model, 
                c_puct=self.c_puct,
                n_simulations=self.n_simulations,
                dirichlet_alpha=self.dirichlet_alpha,
                epsilon=self.epsilon
            )

            for _ in range(n_games_per_iter):
                iteration_examples.extend(best_policy.play_self_game(temp=1.0))

            # Create and train new model
            new_model = copy.deepcopy(best_model)
            new_policy = MancalaModelMCTSPolicy(
                new_model, 
                c_puct=self.c_puct,
                n_simulations=self.n_simulations,
                dirichlet_alpha=self.dirichlet_alpha,
                epsilon=self.epsilon
            )

            new_policy._train_on_examples(
                iteration_examples, 
                batch_size=batch_size, 
                epochs=epochs, 
                lr=lr
            )

            # Evaluate against both previous best and MCTS
            win_rate_vs_old = new_policy.pit(best_policy, n_games=pit_games)
            print(f"New model vs old model win rate = {win_rate_vs_old*100:.2f}%")
            
            mcts_win_rate = new_policy.pit(mcts_opponent, n_games=pit_games, opponent_is_mcts=True)
            print(f"New model vs MCTS win rate = {mcts_win_rate*100:.2f}%")
            print(f"Previous best vs MCTS win rate = {best_mcts_win_rate*100:.2f}%")

            # More flexible acceptance criteria
            should_accept = False
            
            # Accept if significantly better against MCTS
            if mcts_win_rate >= best_mcts_win_rate + 0.1:  # 10% improvement threshold
                print("New model shows significant improvement against MCTS -> Accepting")
                should_accept = True
            # Accept if better against MCTS and competitive with old model
            elif mcts_win_rate >= best_mcts_win_rate and win_rate_vs_old >= 0.45:
                print("New model improves vs MCTS while maintaining reasonable performance vs old model -> Accepting")
                should_accept = True
            # Accept if significantly better against old model
            elif win_rate_vs_old >= threshold + 0.04 and mcts_win_rate >= best_mcts_win_rate - 0.04:  # 10% above threshold
                print("New model shows significant improvement against old model -> Accepting")
                should_accept = True
            elif win_rate_vs_old >= threshold + 0.08 and mcts_win_rate >= best_mcts_win_rate - 0.08:
                print("New model shows significant improvement against old model -> Accepting")
                should_accept = True
            elif win_rate_vs_old >= threshold + 0.14:
                print("New model shows significant improvement against old model and MCTS -> Accepting")
                should_accept = True

            if should_accept:
                best_model = copy.deepcopy(new_model)
                best_mcts_win_rate = mcts_win_rate
                torch.save(best_model.state_dict(), f"./modelsv4/policy_model_iter_{iteration+1000}.pth")
            else:
                torch.save(best_model.state_dict(), f"./modelsv4/policy_model_iter_{iteration+100}.pth")
                print("New model did not meet acceptance criteria -> Keeping old model.")

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