import random
import torch
import torch.optim as optim
from MancalaCNN import MancalaCNN
from engine import MancalaGame
# import torch.nn.functional as F

model_p1 = MancalaCNN()
model_p2 = MancalaCNN()

optimizer_p1 = optim.Adam(model_p1.parameters(), lr=0.001)
optimizer_p2 = optim.Adam(model_p2.parameters(), lr=0.001)

criterion = torch.nn.CrossEntropyLoss()

def self_play_game(model_p1, model_p2):
    game = MancalaGame()
    history = []

    print("Game started!")

    while not game.is_game_over():
        current_player = game.get_current_player()
        valid_moves = game.get_valid_moves()

        # print(f"Player {current_player} is making a move. Valid moves: {valid_moves}")

        p1_side = torch.tensor(game.board[0:6], dtype=torch.float32).unsqueeze(0)
        p2_side = torch.tensor(game.board[7:13], dtype=torch.float32).unsqueeze(0)

        inputs = torch.stack((p1_side, p2_side), dim=0).unsqueeze(0)

        epsilon = 0.1

        if current_player == 1:
            
            with torch.no_grad():
                move_scores = model_p1(inputs)
            
            if random.random() < epsilon:
                predicted_move = random.choice(valid_moves)
            else:
                for move in range(move_scores.shape[-1]):
                    if move not in valid_moves:
                        move_scores[0, move] = float('-inf')  # Mask invalid moves

                predicted_move = torch.argmax(move_scores).item()

            history.append((inputs, predicted_move, current_player))
        else:
            with torch.no_grad():
                move_scores = model_p2(inputs)

            if random.random() < epsilon:
                predicted_move = random.choice(valid_moves)
            else:
                for move in range(move_scores.shape[-1]):
                    if move + 7 not in valid_moves:
                        move_scores[0, move] = float('-inf')  # Mask invalid moves

                predicted_move = torch.argmax(move_scores).item() + 7

            history.append((inputs, predicted_move, current_player))

        game.make_move(predicted_move)
        # print(f"Player {current_player} made move: {predicted_move}")

    p1_score, p2_score = game.get_score()
    if p1_score > p2_score:
        winner = 1
    elif p2_score > p1_score:
        winner = 2
    else:
        winner = 0

    # Compute rewards (simple reward: +1 for win, -1 for loss)
    # rewards = []
    # for _, _, player in history:
    #     if winner == 1 and player == 1:
    #         rewards.append(1)
    #     elif winner == 2 and player == 2:
    #         rewards.append(2)
    #     elif winner == 0:
    #         rewards.append(0)
    #     else:
    #         rewards.append(0)
    reward = winner

    print(reward)
    return history, reward

print("Training started")

num_epochs = 100
for epoch in range(num_epochs):
    print(f"Starting epoch {epoch + 1} of {num_epochs}")
    history, reward = self_play_game(model_p1, model_p2)
    
    for (inputs, predicted_move, current_player) in history:
        if current_player == 1:
            optimizer_p1.zero_grad()
            outputs = model_p1(inputs)
            loss = criterion(outputs, torch.tensor([predicted_move]))
            if reward == 1:
                loss = loss * 1
            elif reward == 2:
                loss = loss * -1
            else:
                loss = loss * 0
            loss.backward()
            optimizer_p1.step()

        if current_player == 2:
            optimizer_p2.zero_grad()
            outputs = model_p2(inputs)

            target_move = predicted_move - 7

            loss = criterion(outputs, torch.tensor([target_move]))
            if reward == 1:
                loss = loss * -1
            elif reward == 2:
                loss = loss * 1
            else:
                loss = loss * 0
            loss.backward()
            optimizer_p2.step()

    print(f"Epoch {epoch + 1}/{num_epochs} completed")

print("Training complete")

torch.save(model_p1.state_dict(), 'mancala_model_p1.pth')
torch.save(model_p2.state_dict(), 'mancala_model_p2.pth')

def evaluate_models(model_p1, model_p2, num_games=10):
    model_p1.eval()
    model_p2.eval()

    wins_p1 = 0
    wins_p2 = 0
    draws = 0

    for _ in range(num_games):
        game = MancalaGame()
        history, reward = self_play_game(model_p1, model_p2)
        if reward == 1:
            wins_p1 += 1
        elif reward == 2:
            wins_p2 += 1
        else:
            draws += 1

    print(f"After {num_games} games:")
    print(f"Player 1 wins: {wins_p1}")
    print(f"Player 2 wins: {wins_p2}")
    print(f"Draws: {draws}")

evaluate_models(model_p1, model_p2, num_games=100)
