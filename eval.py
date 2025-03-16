import torch
import random
from MancalaModel import MancalaModelMCTS, MancalaModel
from engine import MancalaGame

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

if __name__ == "__main__":
    model = MancalaModel()
    model.load_state_dict(torch.load('mancala_model.pth', map_location='cpu'))
    model.eval()
    evaluate_mcts_vs_nn(model, num_games=100, mcts_simulations=50)