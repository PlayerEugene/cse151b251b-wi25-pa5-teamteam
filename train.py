import random
import torch
import torch.optim as optim
from MancalaModel import MancalaModel
from engine import MancalaGame
from torch.optim.lr_scheduler import StepLR

model = MancalaModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# scheduler = StepLR(optimizer, step_size=100, gamma=0.5)
criterion = torch.nn.CrossEntropyLoss()

def self_play_game(model):
    game = MancalaGame()
    history = []

    # print("Game started!")

    while not game.is_game_over():
        current_player = game.get_current_player()
        valid_moves = game.get_valid_moves()

        # print(f"Player {current_player} is making a move. Valid moves: {valid_moves}")

        inputs = torch.tensor((game.board[0:6] + game.board[7:13] + [current_player]), dtype=torch.float32)

        # epsilon = 0.1
        epsilon = max(0.1, 1.0 - (epoch / num_epochs))

        with torch.no_grad():
            move_scores, state_value = model(inputs)
        
        # Mask invalid moves (set their scores to -inf)
        for move in range(move_scores.shape[-1]):
            if current_player == 1:
            # if move < 6:
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

# print("Training started")
num_epochs = 10000
'''EXPLORE GAMMA REWARD DECAY'''
gamma = 0.9
for epoch in range(num_epochs):
    print(f"Starting epoch {epoch + 1} of {num_epochs}")
    history, winner = self_play_game(model)
    
    # for (inputs, predicted_move, current_player) in reversed(history):
    optimizer.zero_grad()

    inputs, predicted_move, current_player = zip(*history)

    inputs = torch.stack(inputs)
    predicted_move = list(predicted_move)
    predicted_move = [move - 1 if move >= 7 else move for move in predicted_move]
    predicted_move = torch.tensor(predicted_move, dtype=torch.long)
    current_player = list(current_player)

    move_scores, state_value = model(inputs)
    cross_entropy = torch.nn.CrossEntropyLoss()
    mean_squared = torch.nn.MSELoss()

    reward = [1 if winner == player else 0 if winner == 0 else -1 for player in current_player]

    # Apply the reward to the model's predicted action
    # Policy loss: Using the softmax output of the model
    # policy_loss = torch.log(torch.softmax(move_scores, dim=-1)[predicted_move]) * reward
    one_hot_moves = torch.nn.functional.one_hot(predicted_move, num_classes=12).float()

    policy_loss = cross_entropy(move_scores, one_hot_moves)

    # print(predicted_move)
    # print(current_player)

    # Value loss: Compare the predicted state value to the total reward
    # value_loss = (state_value - reward) ** 2
    value_loss = mean_squared(state_value, torch.tensor(reward).unsqueeze(1).float())

    loss = policy_loss + value_loss
    # loss = torch.cat((policy_loss.unsqueeze(0), value_loss.unsqueeze(0)), dim=-1)

    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    loss.backward()
    optimizer.step()

    # scheduler.step()

    print(f"Epoch {epoch + 1}/{num_epochs} completed")

# print("Training complete")

torch.save(model.state_dict(), 'mancala_model.pth')

def evaluate_models(model, num_games=100):
    model.eval()

    wins_p1 = 0
    wins_p2 = 0
    draws = 0

    for _ in range(num_games):
        game = MancalaGame()
        history, winner = self_play_game(model)
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

evaluate_models(model, num_games=1000)
