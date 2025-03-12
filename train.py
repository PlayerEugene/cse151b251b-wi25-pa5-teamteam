import random
import torch
import torch.optim as optim
from MancalaCNN import MancalaCNN
from engine import MancalaGame

model = MancalaCNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

def self_play_game(model):
    game = MancalaGame()
    history = []

    print("Game started!")

    while not game.is_game_over():
        current_player = game.get_current_player()
        valid_moves = game.get_valid_moves()

        # print(f"Player {current_player} is making a move. Valid moves: {valid_moves}")

        p1_side = torch.tensor(game.board[0:6], dtype=torch.float32).unsqueeze(0)
        p2_side = torch.tensor(game.board[7:13], dtype=torch.float32).unsqueeze(0)

        player_turn = torch.tensor([1.0 if current_player == 1 else -1.0] * 6, dtype=torch.float32).unsqueeze(0)

        inputs = torch.cat((p1_side, p2_side, player_turn), dim=1)
        inputs = inputs.view(1, 3, 1, 6)

        epsilon = 0.1

        with torch.no_grad():
            move_scores = model(inputs)
        
        # Mask invalid moves (set their scores to -inf)
        for move in range(move_scores.shape[-1]):
            if move not in valid_moves:
                move_scores[0, move] = float('-inf')

        if random.random() < epsilon:
            predicted_move = random.choice(valid_moves)
        else:
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

    return history, winner

print("Training started")

num_epochs = 100
for epoch in range(num_epochs):
    print(f"Starting epoch {epoch + 1} of {num_epochs}")
    history, winner = self_play_game(model)
    
    for (inputs, predicted_move, current_player) in history:
        optimizer.zero_grad()
        outputs = model(inputs)

        if (predicted_move >= 7):
            predicted_move -= 1

        loss = criterion(outputs, torch.tensor([predicted_move]))

        if winner == 1 and current_player == 1:
            loss = loss * 1
        elif winner == 2 and current_player == 2:
            loss = loss * 1
        elif winner == 1 and current_player == 2:
            loss = loss * -1
        elif winner == 2 and current_player == 1:
            loss = loss * -1
        else:
            loss = loss * 0

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs} completed")

print("Training complete")

torch.save(model.state_dict(), 'mancala_model.pth')

def evaluate_models(model, num_games=10):
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

evaluate_models(model, num_games=100)
