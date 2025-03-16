# evaluate.py

import torch
import random
from matplotlib import pyplot as plt
from MancalaModel import MancalaModel
from engine import MancalaGame

# Load the pre-trained model
def load_model(model_path='mancala_model.pth'):
    model = MancalaModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def self_play_game(model, epoch, num_epochs):
    game = MancalaGame()
    history = []

    while not game.is_game_over():
        current_player = game.get_current_player()
        valid_moves = game.get_valid_moves()

        inputs = torch.tensor((game.board[0:6] + game.board[7:13] + [current_player]), dtype=torch.float32)

        epsilon = max(0.1, 1.0 - (epoch / num_epochs))

        with torch.no_grad():
            move_scores, state_value = model(inputs)
        
        # Mask invalid moves (set their scores to -inf)
        for move in range(move_scores.shape[-1]):
            if current_player == 1:
                if move not in valid_moves:
                    move_scores[move] = float(0)
            else:
                if move + 1 not in valid_moves:
                    move_scores[move] = float(0)

        if random.random() < epsilon:
            predicted_move = random.choice(valid_moves)
        else:
            predicted_move = torch.argmax(move_scores).item()
            if predicted_move >= 6:
                predicted_move += 1

        history.append([inputs, predicted_move, current_player])

        game.make_move(predicted_move)

    p1_score, p2_score = game.get_score()
    if p1_score > p2_score:
        winner = 1
    elif p2_score > p1_score:
        winner = 2
    else:
        winner = 0

    return history, winner

def evaluate_models(model, num_games=100):
    model.eval()

    wins_p1 = 0
    wins_p2 = 0
    draws = 0

    for epoch in range(num_games):
        game = MancalaGame()
        history, winner = self_play_game(model, epoch, num_games)
        if winner == 1:
            wins_p1 += 1
        elif winner == 2:
            wins_p2 += 1
        else:
            draws += 1

    print(f"After {num_games} games:")
    print(f"Player 1 wins: {wins_p1}")
    print(f"Player 2 wins: {wins_p2}")
    print(f"Draws: {draws}")

    return wins_p1, wins_p2, draws

def evaluate_multiple_runs(model, num_runs=10, num_games=100):
    results_p1 = []
    results_p2 = []
    results_draws = []

    for _ in range(num_runs):
        wins_p1, wins_p2, draws = evaluate_models(model, num_games)
        results_p1.append(wins_p1)
        results_p2.append(wins_p2)
        results_draws.append(draws)

    return results_p1, results_p2, results_draws

def plot_results(num_runs, results_p1, results_p2, results_draws):
    plt.figure(figsize=(10, 6))

    plt.plot(range(1, num_runs + 1), results_p1, label="Player 1 Wins", marker='o', linestyle='-', color='b')
    plt.plot(range(1, num_runs + 1), results_p2, label="Player 2 Wins", marker='o', linestyle='-', color='r')
    plt.plot(range(1, num_runs + 1), results_draws, label="Draws", marker='o', linestyle='-', color='g')

    plt.title(f"Model Performance Over {num_runs} Runs")
    plt.xlabel('Run Number')
    plt.ylabel('Number of Games')
    plt.legend()

    plt.show()

# Main function to load the model, evaluate, and plot
def evaluate():
    model = load_model('mancala_model.pth')  # Load the saved model
    
    # Run the evaluation for multiple runs and games
    num_runs = 10
    num_games = 1000
    results_p1, results_p2, results_draws = evaluate_multiple_runs(model, num_runs=num_runs, num_games=num_games)
    
    # Plot the results
    plot_results(num_runs, results_p1, results_p2, results_draws)

if __name__ == "__main__":
    evaluate()
