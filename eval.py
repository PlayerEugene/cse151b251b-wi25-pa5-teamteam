#from matplotlib import pyplot as plt
import torch
import random
from MancalaModel import MancalaModelMCTS, MancalaModel, MancalaModelMCTSPolicy, BaseMancalaModel, MancalaModelv2
from engine import MancalaGame
from train_mctspolicy import evaluate_mctspolicy_vs_mcts
import os

def evaluate_mcts_vs_nn(model, num_games=10, mcts_simulations=50):
    """
    Evaluates MCTS vs. NN, randomly assigning MCTS to either P1 or P2 each game,
    with additional logging for debugging.
    Also tracks first player vs second player advantage for each agent type.
    """
    model.eval()
    
    mcts_wins = 0
    nn_wins = 0
    draws = 0
    
    p1_wins = 0
    p2_wins = 0
    
    mcts_as_p1_games = 0
    mcts_as_p1_wins = 0
    mcts_as_p2_games = 0
    mcts_as_p2_wins = 0
    
    nn_as_p1_games = 0
    nn_as_p1_wins = 0
    nn_as_p2_games = 0
    nn_as_p2_wins = 0

    for game_idx in range(num_games):
        print(f"\n=== Starting Game {game_idx + 1}/{num_games} ===")

        game = MancalaGame()
        
        mcts_is_p1 = random.choice([True, False])
        print(f"MCTS is playing as Player {1 if mcts_is_p1 else 2}")
        
        if mcts_is_p1:
            mcts_as_p1_games += 1
            nn_as_p2_games += 1
        else:
            mcts_as_p2_games += 1
            nn_as_p1_games += 1

        if mcts_is_p1:
            mcts_agent = MancalaModelMCTS(num_simulations=mcts_simulations,
                                          ucb_c=1.4, mcts_player=1)
        else:
            mcts_agent = MancalaModelMCTS(num_simulations=mcts_simulations,
                                          ucb_c=1.4, mcts_player=2)
        
        move_count = 0
        while not game.is_game_over():
            current_player = game.get_current_player()
            move_count += 1
            
            if current_player == mcts_agent.mcts_player:
                best_move = mcts_agent.mcts(game)
                game.make_move(best_move)
            else:
                valid_moves = game.get_valid_moves()
                inputs = torch.tensor(
                    game.board[0:6] + game.board[7:13] + [current_player],
                    dtype=torch.float32
                )

                with torch.no_grad():
                    move_scores, state_value = model(inputs)

                valid_nn_moves = []
                for move in valid_moves:
                    if current_player == 1:
                        valid_nn_moves.append(move)
                    else:
                        valid_nn_moves.append(move - 7)
                
                masked_scores = move_scores.clone()
                for move_idx in range(masked_scores.shape[-1]):
                    if move_idx not in valid_nn_moves:
                        masked_scores[move_idx] = float('-inf')

                predicted_nn_move = torch.argmax(masked_scores).item()
                
                if current_player == 1:
                    game_move = predicted_nn_move
                else:
                    game_move = predicted_nn_move + 7
                
                game.make_move(game_move)

                # for move in range(move_scores.shape[-1]):
                #     if current_player == 1:
                #     # if move < 6:
                #         if move not in valid_moves:
                #             move_scores[move] = float(0)
                #     else:
                #         if move + 1 not in valid_moves:
                #             move_scores[move] = float(0)
                # predicted_move = torch.argmax(move_scores).item()
                # if predicted_move >= 6:
                #     predicted_move += 1

                # game.make_move(predicted_move)
        
        p1_score, p2_score = game.get_score()
        if p1_score > p2_score:
            p1_wins += 1
            if mcts_is_p1:
                mcts_wins += 1
                mcts_as_p1_wins += 1
                winner_str = "MCTS (Player 1)"
            else:
                nn_wins += 1
                nn_as_p1_wins += 1
                winner_str = "NN (Player 1)"
        elif p2_score > p1_score:
            p2_wins += 1
            if mcts_is_p1:
                nn_wins += 1
                nn_as_p2_wins += 1
                winner_str = "NN (Player 2)"
            else:
                mcts_wins += 1
                mcts_as_p2_wins += 1
                winner_str = "MCTS (Player 2)"
        else:
            draws += 1
            winner_str = "Tie"
        
        print(f"Game {game_idx + 1}/{num_games} completed: "
              f"Player1={p1_score} vs Player2={p2_score} | Winner: {winner_str}")

    print("\n--- MCTS vs Neural Net Results ---")
    print(f"Total Games: {num_games}")
    print(f"MCTS wins: {mcts_wins} ({mcts_wins/num_games*100:.1f}%)")
    print(f"NN wins: {nn_wins} ({nn_wins/num_games*100:.1f}%)")
    print(f"Draws: {draws} ({draws/num_games*100:.1f}%)")
    print("----------------------------------")
    
    print("\n--- First vs Second Player Advantage ---")
    print(f"Player 1 (first) wins: {p1_wins} ({p1_wins/num_games*100:.1f}%)")
    print(f"Player 2 (second) wins: {p2_wins} ({p2_wins/num_games*100:.1f}%)")
    print(f"Draws: {draws} ({draws/num_games*100:.1f}%)")
    print("----------------------------------")
    
    print("\n--- Detailed Agent Position Stats ---")
    print("MCTS as Player 1:")
    if mcts_as_p1_games > 0:
        print(f"  Games: {mcts_as_p1_games}")
        print(f"  Wins: {mcts_as_p1_wins} ({mcts_as_p1_wins/mcts_as_p1_games*100:.1f}%)")
    else:
        print("  No games played")
        
    print("MCTS as Player 2:")
    if mcts_as_p2_games > 0:
        print(f"  Games: {mcts_as_p2_games}")
        print(f"  Wins: {mcts_as_p2_wins} ({mcts_as_p2_wins/mcts_as_p2_games*100:.1f}%)")
    else:
        print("  No games played")
        
    print("NN as Player 1:")
    if nn_as_p1_games > 0:
        print(f"  Games: {nn_as_p1_games}")
        print(f"  Wins: {nn_as_p1_wins} ({nn_as_p1_wins/nn_as_p1_games*100:.1f}%)")
    else:
        print("  No games played")
        
    print("NN as Player 2:")
    if nn_as_p2_games > 0:
        print(f"  Games: {nn_as_p2_games}")
        print(f"  Wins: {nn_as_p2_wins} ({nn_as_p2_wins/nn_as_p2_games*100:.1f}%)")
    else:
        print("  No games played")
    print("----------------------------------")

    return mcts_wins, nn_wins, draws

def run_multiple_evaluations(model, num_runs=10, num_games_per_run=1000, mcts_simulations=500):
    mcts_wins_all_runs = []
    nn_wins_all_runs = []
    draws_all_runs = []

    for run_idx in range(num_runs):
        print(f"\nStarting evaluation run {run_idx + 1}/{num_runs}...")
        mcts_wins, nn_wins, draws = evaluate_mcts_vs_nn(model, num_games=num_games_per_run, mcts_simulations=mcts_simulations)
        mcts_wins_all_runs.append(mcts_wins)
        nn_wins_all_runs.append(nn_wins)
        draws_all_runs.append(draws)

    return mcts_wins_all_runs, nn_wins_all_runs, draws_all_runs


def plot_results(mcts_wins_all_runs, nn_wins_all_runs, draws_all_runs):
    run_numbers = list(range(1, len(mcts_wins_all_runs) + 1))

    plt.figure(figsize=(10, 6))

    plt.plot(run_numbers, mcts_wins_all_runs, label="MCTS Wins", marker='o')
    plt.plot(run_numbers, nn_wins_all_runs, label="NN Wins", marker='o')
    plt.plot(run_numbers, draws_all_runs, label="Draws", marker='o')

    plt.xlabel("Evaluation Run")
    plt.ylabel("Number of Outcomes")
    plt.title("MCTS vs NN Evaluation Results")
    plt.legend()
    plt.grid(True)

    plt.show()

def evaluate_nn_vs_mctspolicy(model, policy: MancalaModelMCTSPolicy, num_games=10):
    """
    Pit a neural network model against a MancalaModelMCTSPolicy.
    Randomly assign which side each agent plays (P1 or P2) each game.
    Track and print detailed statistics.
    """
    model.eval()
    
    policy_wins = 0
    nn_wins = 0
    draws = 0
    
    p1_wins = 0
    p2_wins = 0
    
    policy_as_p1_games = 0
    policy_as_p1_wins = 0
    policy_as_p2_games = 0
    policy_as_p2_wins = 0
    
    nn_as_p1_games = 0
    nn_as_p1_wins = 0
    nn_as_p2_games = 0
    nn_as_p2_wins = 0

    for game_idx in range(num_games):
        game = MancalaGame()
        
        policy_is_p1 = game_idx % 2 == 0  # Alternate who plays first
        
        if policy_is_p1:
            policy_as_p1_games += 1
            nn_as_p2_games += 1
        else:
            policy_as_p2_games += 1
            nn_as_p1_games += 1

        while not game.is_game_over():
            current_player = game.get_current_player()
            
            if (current_player == 1 and policy_is_p1) or (current_player == 2 and not policy_is_p1):
                # Policy's turn
                policy_dist = policy.get_action_prob(game, temp=0.0, add_dirichlet_noise=False)
                best_action_idx = max(range(len(policy_dist)), key=lambda i: policy_dist[i])
                best_action = best_action_idx if best_action_idx < 6 else best_action_idx + 1
                game.make_move(best_action)
            else:
                # Neural Network's turn
                valid_moves = game.get_valid_moves()
                inputs = torch.tensor(
                    game.board[0:6] + game.board[7:13] + [current_player],
                    dtype=torch.float32
                )

                with torch.no_grad():
                    move_scores, _ = model(inputs)

                valid_nn_moves = []
                for move in valid_moves:
                    if current_player == 1:
                        valid_nn_moves.append(move)
                    else:
                        valid_nn_moves.append(move - 7)
                
                masked_scores = move_scores.clone()
                for move_idx in range(masked_scores.shape[-1]):
                    if move_idx not in valid_nn_moves:
                        masked_scores[move_idx] = float('-inf')

                predicted_nn_move = torch.argmax(masked_scores).item()
                
                if current_player == 1:
                    game_move = predicted_nn_move
                else:
                    game_move = predicted_nn_move + 7
                
                game.make_move(game_move)
        
        p1_score, p2_score = game.get_score()
        if p1_score > p2_score:
            p1_wins += 1
            if policy_is_p1:
                policy_wins += 1
                policy_as_p1_wins += 1
            else:
                nn_wins += 1
                nn_as_p1_wins += 1
        elif p2_score > p1_score:
            p2_wins += 1
            if policy_is_p1:
                nn_wins += 1
                nn_as_p2_wins += 1
            else:
                policy_wins += 1
                policy_as_p2_wins += 1
        else:
            draws += 1

    print("\n--- Neural Net vs MCTS Policy Results ---")
    print(f"Total Games: {num_games}")
    print(f"Policy wins: {policy_wins} ({policy_wins/num_games*100:.1f}%)")
    print(f"NN wins: {nn_wins} ({nn_wins/num_games*100:.1f}%)")
    print(f"Draws: {draws} ({draws/num_games*100:.1f}%)")
    
    print("\n--- First vs Second Player Advantage ---")
    print(f"Player 1 (first) wins: {p1_wins} ({p1_wins/num_games*100:.1f}%)")
    print(f"Player 2 (second) wins: {p2_wins} ({p2_wins/num_games*100:.1f}%)")
    print(f"Draws: {draws} ({draws/num_games*100:.1f}%)")
    
    print("\n--- Detailed Agent Position Stats ---")
    print("Policy as Player 1:")
    if policy_as_p1_games > 0:
        print(f"  Games: {policy_as_p1_games}")
        print(f"  Wins: {policy_as_p1_wins} ({policy_as_p1_wins/policy_as_p1_games*100:.1f}%)")
    else:
        print("  No games played")
        
    print("Policy as Player 2:")
    if policy_as_p2_games > 0:
        print(f"  Games: {policy_as_p2_games}")
        print(f"  Wins: {policy_as_p2_wins} ({policy_as_p2_wins/policy_as_p2_games*100:.1f}%)")
    else:
        print("  No games played")
        
    print("NN as Player 1:")
    if nn_as_p1_games > 0:
        print(f"  Games: {nn_as_p1_games}")
        print(f"  Wins: {nn_as_p1_wins} ({nn_as_p1_wins/nn_as_p1_games*100:.1f}%)")
    else:
        print("  No games played")
        
    print("NN as Player 2:")
    if nn_as_p2_games > 0:
        print(f"  Games: {nn_as_p2_games}")
        print(f"  Wins: {nn_as_p2_wins} ({nn_as_p2_wins/nn_as_p2_games*100:.1f}%)")
    else:
        print("  No games played")
    print("----------------------------------")

    return policy_wins, nn_wins, draws

def evaluate_mctspolicy_vs_self(policy: MancalaModelMCTSPolicy, num_games=10):
    """
    Pit a MancalaModelMCTSPolicy against itself.
    Track and print detailed statistics about first vs second player advantage.
    """
    p1_wins = 0
    p2_wins = 0
    draws = 0
    
    for game_idx in range(num_games):
        game = MancalaGame()
        
        while not game.is_game_over():
            current_player = game.get_current_player()
            
            # Both players use the same policy
            policy_dist = policy.get_action_prob(game, temp=0.0, add_dirichlet_noise=False)
            best_action_idx = max(range(len(policy_dist)), key=lambda i: policy_dist[i])
            best_action = best_action_idx if best_action_idx < 6 else best_action_idx + 1
            game.make_move(best_action)
        
        p1_score, p2_score = game.get_score()
        if p1_score > p2_score:
            p1_wins += 1
        elif p2_score > p1_score:
            p2_wins += 1
        else:
            draws += 1

    print("\n--- MCTS Policy Self-Play Results ---")
    print(f"Total Games: {num_games}")
    print(f"Player 1 wins: {p1_wins} ({p1_wins/num_games*100:.1f}%)")
    print(f"Player 2 wins: {p2_wins} ({p2_wins/num_games*100:.1f}%)")
    print(f"Draws: {draws} ({draws/num_games*100:.1f}%)")
    print("----------------------------------")

    return p1_wins, p2_wins, draws

def evaluate_mctspolicy_vs_random(policy: MancalaModelMCTSPolicy, num_games=10):
    """
    Pit a MancalaModelMCTSPolicy against a random player.
    Randomly assign which side each agent plays (P1 or P2) each game.
    Track and print detailed statistics.
    """
    policy_wins = 0
    random_wins = 0
    draws = 0
    
    p1_wins = 0
    p2_wins = 0
    
    policy_as_p1_games = 0
    policy_as_p1_wins = 0
    policy_as_p2_games = 0
    policy_as_p2_wins = 0
    
    random_as_p1_games = 0
    random_as_p1_wins = 0
    random_as_p2_games = 0
    random_as_p2_wins = 0

    for game_idx in range(num_games):
        game = MancalaGame()
        
        policy_is_p1 = game_idx % 2 == 0  # Alternate who plays first
        
        if policy_is_p1:
            policy_as_p1_games += 1
            random_as_p2_games += 1
        else:
            policy_as_p2_games += 1
            random_as_p1_games += 1

        while not game.is_game_over():
            current_player = game.get_current_player()
            valid_moves = game.get_valid_moves()
            
            if (current_player == 1 and policy_is_p1) or (current_player == 2 and not policy_is_p1):
                # Policy's turn
                policy_dist = policy.get_action_prob(game, temp=0.0, add_dirichlet_noise=False)
                best_action_idx = max(range(len(policy_dist)), key=lambda i: policy_dist[i])
                best_action = best_action_idx if best_action_idx < 6 else best_action_idx + 1
                game.make_move(best_action)
            else:
                # Random player's turn
                random_move = random.choice(valid_moves)
                game.make_move(random_move)
        
        p1_score, p2_score = game.get_score()
        if p1_score > p2_score:
            p1_wins += 1
            if policy_is_p1:
                policy_wins += 1
                policy_as_p1_wins += 1
            else:
                random_wins += 1
                random_as_p1_wins += 1
        elif p2_score > p1_score:
            p2_wins += 1
            if policy_is_p1:
                random_wins += 1
                random_as_p2_wins += 1
            else:
                policy_wins += 1
                policy_as_p2_wins += 1
        else:
            draws += 1

    print("\n--- MCTS Policy vs Random Results ---")
    print(f"Total Games: {num_games}")
    print(f"Policy wins: {policy_wins} ({policy_wins/num_games*100:.1f}%)")
    print(f"Random wins: {random_wins} ({random_wins/num_games*100:.1f}%)")
    print(f"Draws: {draws} ({draws/num_games*100:.1f}%)")
    
    print("\n--- First vs Second Player Advantage ---")
    print(f"Player 1 (first) wins: {p1_wins} ({p1_wins/num_games*100:.1f}%)")
    print(f"Player 2 (second) wins: {p2_wins} ({p2_wins/num_games*100:.1f}%)")
    print(f"Draws: {draws} ({draws/num_games*100:.1f}%)")
    
    print("\n--- Detailed Agent Position Stats ---")
    print("Policy as Player 1:")
    if policy_as_p1_games > 0:
        print(f"  Games: {policy_as_p1_games}")
        print(f"  Wins: {policy_as_p1_wins} ({policy_as_p1_wins/policy_as_p1_games*100:.1f}%)")
    else:
        print("  No games played")
        
    print("Policy as Player 2:")
    if policy_as_p2_games > 0:
        print(f"  Games: {policy_as_p2_games}")
        print(f"  Wins: {policy_as_p2_wins} ({policy_as_p2_wins/policy_as_p2_games*100:.1f}%)")
    else:
        print("  No games played")
        
    print("Random as Player 1:")
    if random_as_p1_games > 0:
        print(f"  Games: {random_as_p1_games}")
        print(f"  Wins: {random_as_p1_wins} ({random_as_p1_wins/random_as_p1_games*100:.1f}%)")
    else:
        print("  No games played")
        
    print("Random as Player 2:")
    if random_as_p2_games > 0:
        print(f"  Games: {random_as_p2_games}")
        print(f"  Wins: {random_as_p2_wins} ({random_as_p2_wins/random_as_p2_games*100:.1f}%)")
    else:
        print("  No games played")
    print("----------------------------------")

    return policy_wins, random_wins, draws

if __name__ == "__main__":
    policy_model_path = '/workspace/cse151b251b-wi25-pa5-teamteam/modelsv4/policy_model_iter_106.pth'
    nn_model_path = 'mancala_model_nn1.pth'
    # Load both models
    nn_model = BaseMancalaModel()
    nn_model.load_state_dict(torch.load(nn_model_path))
    nn_model.eval()

    policy_model = MancalaModelv2() 
    policy_model.load_state_dict(torch.load(policy_model_path))
    policy_model.eval()

    # Create MCTS policy from policy model
    mcts_policy = MancalaModelMCTSPolicy(
        model=policy_model,
        c_puct=1.4,
        n_simulations=50,
        dirichlet_alpha=0.03,
        epsilon=0.25
    )

    print("\nEvaluating Base NN vs MCTS Policy...")
    evaluate_nn_vs_mctspolicy(nn_model, mcts_policy, num_games=100)

    print("\nEvaluating MCTS Policy vs MCTS...")
    evaluate_mctspolicy_vs_mcts(mcts_policy, num_games=100, mcts_simulations=50)

    print("\n Evaluating MCTS Policy vs itself...")
    evaluate_mctspolicy_vs_self(mcts_policy, num_games=100)

    print("\nEvaluating MCTS Policy vs Random...")
    evaluate_mctspolicy_vs_random(mcts_policy, num_games=100)
    # # Now evaluate the regular model
    # print("\nEvaluating Neural Network vs MCTS...")
    # model = MancalaModel()
    # model.load_state_dict(torch.load('mancala_model.pth'))
    # model.eval()

    # # evaluate_mcts_vs_nn(model, num_games=1000, mcts_simulations=500)
    # num_runs = 10
    # num_games_per_run = 100

    # mcts_wins_all_runs, nn_wins_all_runs, draws_all_runs = run_multiple_evaluations(
    #     model, num_runs=num_runs, num_games_per_run=num_games_per_run, mcts_simulations=500)
    # plot_results(mcts_wins_all_runs, nn_wins_all_runs, draws_all_runs)
