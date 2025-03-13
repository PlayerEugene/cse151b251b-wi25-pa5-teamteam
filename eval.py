import torch
from MancalaModel import MancalaModel, MancalaModelMCTS
from engine import MancalaGame
import random

def evaluate_mcts_vs_nn(model, num_games=10, mcts_simulations=500):
    model.eval()

    mcts_agent = MancalaModelMCTS(num_simulations=mcts_simulations, ucb_c=1.4)

    mcts_wins = 0
    nn_wins = 0
    draws = 0

    for game_idx in range(num_games):
        game = MancalaGame()

        while not game.is_game_over():
            current_player = game.get_current_player()

            if current_player == 1:
                best_move = mcts_agent.mcts(game)
                game.make_move(best_move)

            else:
                valid_moves = game.get_valid_moves()
                inputs = torch.tensor(game.board[0:6] + game.board[7:13] + [current_player],
                                      dtype=torch.float32)

                with torch.no_grad():
                    move_scores, state_value = model(inputs)

                for move_idx in range(move_scores.shape[-1]):
                    if (move_idx + 1) not in valid_moves:
                        move_scores[move_idx] = float('-inf')

                predicted_move = torch.argmax(move_scores).item()
                if predicted_move >= 6:
                    predicted_move += 1

                game.make_move(predicted_move)

        p1_score, p2_score = game.get_score()
        if p1_score > p2_score:
            mcts_wins += 1
        elif p2_score > p1_score:
            nn_wins += 1
        else:
            draws += 1

        print(f"Game {game_idx + 1}/{num_games} completed: Player1={p1_score} vs Player2={p2_score}")

    print("\n--- MCTS vs Neural Net Results ---")
    print(f"Total Games: {num_games}")
    print(f"MCTS (P1) wins: {mcts_wins}")
    print(f"NN   (P2) wins: {nn_wins}")
    print(f"Draws:          {draws}")
    print("----------------------------------")


if __name__ == "__main__":
    model = MancalaModel()
    model.load_state_dict(torch.load('mancala_model.pth', map_location='cpu'))
    model.eval()

    evaluate_mcts_vs_nn(model, num_games=100, mcts_simulations=500)