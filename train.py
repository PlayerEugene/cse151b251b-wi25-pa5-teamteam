import torch
import torch.optim as optim
from MancalaCNN import MancalaCNN
from engine import MancalaGame
import torch.nn.functional as F

model_p1 = MancalaCNN()
model_p2 = MancalaCNN()

optimizer_p1 = optim.Adam(model_p1.parameters(), lr=0.001)
optimizer_p2 = optim.Adam(model_p2.parameters(), lr=0.001)

criterion = torch.nn.CrossEntropyLoss()

def self_play_game(model_p1, model_p2):
    game = MancalaGame()
    history = []

    while not game.is_game_over():
        current_player = game.get_current_player()
        valid_moves = game.get_valid_moves()

        p1_side = torch.tensor(game.board[0:6], dtype=torch.float32).unsqueeze(0)
        p2_side = torch.tensor(game.board[7:13], dtype=torch.float32).unsqueeze(0)

        if current_player == 1:
            inputs = torch.stack((p1_side, p2_side), dim=0).unsqueeze(0)
            
            with torch.no_grad():
                move_scores = model_p1(inputs)
            predicted_move = torch.argmax(move_scores).item()

            history.append((inputs, predicted_move, current_player))
        else:
            inputs = torch.stack((p1_side, p2_side), dim=0).unsqueeze(0)
            
            with torch.no_grad():
                move_scores = model_p2(inputs)
            predicted_move = torch.argmax(move_scores).item()

            history.append((inputs, predicted_move, current_player))

        game.make_move(predicted_move)

    p1_score, p2_score = game.get_score()
    if p1_score > p2_score:
        winner = 1
    elif p2_score > p1_score:
        winner = 2
    else:
        winner = 0

    # Compute rewards (simple reward: +1 for win, -1 for loss)
    rewards = []
    for _, _, player in history:
        if (player == 1 and winner == 1) or (player == 2 and winner == 2):
            rewards.append(1)  # Win
        elif winner == 0:
            rewards.append(0)  # Draw
        else:
            rewards.append(-1)  # Loss

    return history, rewards


num_epochs = 100
for epoch in range(num_epochs):
    history, rewards = self_play_game(model_p1, model_p2)
    
    total_loss = 0
    for (inputs, predicted_move, current_player), reward in zip(history, rewards):
        if current_player == 1:
            optimizer_p1.zero_grad()
            outputs = model_p1(inputs)
            loss = criterion(outputs, torch.tensor([predicted_move]))
            loss.backward()
            optimizer_p1.step()
            total_loss += loss.item()

        if current_player == 2:
            optimizer_p2.zero_grad()
            outputs = model_p2(inputs)
            loss = criterion(outputs, torch.tensor([predicted_move]))
            loss.backward()
            optimizer_p2.step()
            total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs} completed, Loss: {total_loss/len(history):.4f}")

torch.save(model_p1.state_dict(), 'mancala_model_p1.pth')
torch.save(model_p2.state_dict(), 'mancala_model_p2.pth')
