import torch
import os
from MancalaModel import MancalaModel, MancalaModelMCTS, MancalaModelMCTSPolicy
from engine import MancalaGame
import random

def evaluate_mctspolicy_vs_mcts(policy: MancalaModelMCTSPolicy, 
                                num_games=10, 
                                mcts_simulations=50):
    """
    Pit a MancalaModelMCTSPolicy (`policy`) against a pure MCTS agent (MancalaModelMCTS).
    Randomly assign which side the policy plays (P1 or P2) each game,
    and also randomize who starts first.
    Track and print detailed statistics similar to evaluate_mcts_vs_nn.
    """

    policy_wins = 0
    mcts_wins = 0
    draws = 0

    # Tracking first/second player wins overall
    p1_wins = 0
    p2_wins = 0
    
    # Detailed stats for each agent as P1/P2
    policy_as_p1_games = 0
    policy_as_p1_wins = 0
    policy_as_p2_games = 0
    policy_as_p2_wins = 0

    mcts_as_p1_games = 0
    mcts_as_p1_wins = 0
    mcts_as_p2_games = 0
    mcts_as_p2_wins = 0

    for game_idx in range(num_games):
        print(f"\n=== Starting Game {game_idx + 1}/{num_games} ===")
        # Create a fresh game
        game = MancalaGame()
        
        # Randomly decide which side the policy (MancalaModelMCTSPolicy) takes
        policy_is_p1 = random.choice([True, False])
        if policy_is_p1:
            print("Policy is Player 1, MCTS is Player 2")
            # MCTS agent is forced to be player 2
            mcts_agent = MancalaModelMCTS(num_simulations=mcts_simulations,
                                          ucb_c=1.4, 
                                          mcts_player=2)
            policy_as_p1_games += 1
            mcts_as_p2_games += 1
        else:
            print("MCTS is Player 1, Policy is Player 2")
            # MCTS agent is forced to be player 1
            mcts_agent = MancalaModelMCTS(num_simulations=mcts_simulations,
                                          ucb_c=1.4, 
                                          mcts_player=1)
            policy_as_p2_games += 1
            mcts_as_p1_games += 1

        # Randomize who moves first (override default current_player=1)
        starting_player = random.choice([1, 2])
        game.current_player = starting_player
        print(f"Randomly chosen starter: Player {starting_player}")

        while not game.is_game_over():
            current_player = game.get_current_player()

            # Decide whose turn it is, based on current_player and policy_is_p1
            if (current_player == 1 and policy_is_p1) or (current_player == 2 and not policy_is_p1):
                # Policy's turn
                policy_dist = policy.get_action_prob(game, temp=0.0, add_dirichlet_noise=False)
                best_action_idx = max(range(len(policy_dist)), key=lambda i: policy_dist[i])
                # Remember pockets >= 6 for P1 or >= 7 for P2
                # so the mapping is that indices [0..5] map to pockets [0..5],
                # and indices [6..11] map to pockets [7..12].
                best_action = best_action_idx if best_action_idx < 6 else best_action_idx + 1
                
                game.make_move(best_action)
            else:
                # MCTS's turn
                best_move = mcts_agent.mcts(game)
                game.make_move(best_move)
        
        # Game is over; figure out who won
        p1_score, p2_score = game.get_score()
        print(f"Final Board: P1={p1_score}, P2={p2_score}")

        if p1_score > p2_score:
            p1_wins += 1
            winner = 1
            winner_str = "Player 1"
        elif p2_score > p1_score:
            p2_wins += 1
            winner = 2
            winner_str = "Player 2"
        else:
            draws += 1
            winner = 0
            winner_str = "Tie"
        
        # Track agent-specific wins
        if winner == 1:
            if policy_is_p1:
                policy_wins += 1
                policy_as_p1_wins += 1
                print("Winner: Policy (Player 1)")
            else:
                mcts_wins += 1
                mcts_as_p1_wins += 1
                print("Winner: MCTS (Player 1)")
        elif winner == 2:
            if policy_is_p1:
                mcts_wins += 1
                mcts_as_p2_wins += 1
                print("Winner: MCTS (Player 2)")
            else:
                policy_wins += 1
                policy_as_p2_wins += 1
                print("Winner: Policy (Player 2)")
        else:
            print("Result: Tie")

    # Print summary statistics
    print("\n--- Policy vs MCTS Results ---")
    print(f"Total Games: {num_games}")
    print(f"Policy wins: {policy_wins} ({policy_wins/num_games*100:.1f}%)")
    print(f"MCTS wins: {mcts_wins} ({mcts_wins/num_games*100:.1f}%)")
    print(f"Draws: {draws} ({draws/num_games*100:.1f}%)")
    
    print("\n--- First vs Second Player Advantage (Overall) ---")
    print(f"Player 1 (first) wins: {p1_wins} ({p1_wins/num_games*100:.1f}%)")
    print(f"Player 2 (second) wins: {p2_wins} ({p2_wins/num_games*100:.1f}%)")
    print(f"Draws: {draws} ({draws/num_games*100:.1f}%)")
    
    print("\n--- Detailed Agent Position Stats ---")
    # Policy as Player 1
    print("Policy as Player 1:")
    if policy_as_p1_games > 0:
        print(f"  Games: {policy_as_p1_games}")
        print(f"  Wins: {policy_as_p1_wins} ({policy_as_p1_wins/policy_as_p1_games*100:.1f}%)")
    else:
        print("  (No games played as P1)")
    # Policy as Player 2
    print("Policy as Player 2:")
    if policy_as_p2_games > 0:
        print(f"  Games: {policy_as_p2_games}")
        print(f"  Wins: {policy_as_p2_wins} ({policy_as_p2_wins/policy_as_p2_games*100:.1f}%)")
    else:
        print("  (No games played as P2)")

    # MCTS as Player 1
    print("MCTS as Player 1:")
    if mcts_as_p1_games > 0:
        print(f"  Games: {mcts_as_p1_games}")
        print(f"  Wins: {mcts_as_p1_wins} ({mcts_as_p1_wins/mcts_as_p1_games*100:.1f}%)")
    else:
        print("  (No games played as P1)")
    # MCTS as Player 2
    print("MCTS as Player 2:")
    if mcts_as_p2_games > 0:
        print(f"  Games: {mcts_as_p2_games}")
        print(f"  Wins: {mcts_as_p2_wins} ({mcts_as_p2_wins/mcts_as_p2_games*100:.1f}%)")
    else:
        print("  (No games played as P2)")

    print("----------------------------------")

def main():
    model = MancalaModel()

    mcts_policy = MancalaModelMCTSPolicy(
        model=model,
        c_puct=1.4,
        n_simulations=50,
        dirichlet_alpha=0.03,
        epsilon=0.25
    )

    mcts_policy.train_policy_iteration(
        num_iters=20,
        n_games_per_iter=50,
        pit_games=20,
        threshold=0.55,
        batch_size=64,
        epochs=1,
        lr=1e-3
    )

    evaluate_mctspolicy_vs_mcts(mcts_policy, num_games=100, mcts_simulations=50)

    MancalaModelMCTSPolicy.save_policy_weights(mcts_policy, "policy_model.pth")

if __name__ == "__main__":
    main()