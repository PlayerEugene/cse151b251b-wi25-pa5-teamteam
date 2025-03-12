import torch
from MancalaCNN import MancalaCNN
from engine import MancalaGame

model_p1 = MancalaCNN()
model_p2 = MancalaCNN()

model_p1.load_state_dict(torch.load('mancala_model_p1.pth'))
model_p2.load_state_dict(torch.load('mancala_model_p2.pth'))

model_p1.eval()
model_p2.eval()

def play_game():
    game = MancalaGame()
    while not game.is_game_over():
        current_player = game.get_current_player()
        valid_moves = game.get_valid_moves()

        if current_player == 1:
            inputs = torch.tensor(game.get_board_state(), dtype=torch.float32).unsqueeze(0)
            inputs = inputs.permute(0, 3, 1, 2)  # (batch_size, channels, height, width)
            
            with torch.no_grad():
                move_scores = model_p1(inputs)
            predicted_move = torch.argmax(move_scores).item()

        else:
            inputs = torch.tensor(game.get_board_state(), dtype=torch.float32).unsqueeze(0)
            inputs = inputs.permute(0, 3, 1, 2)
            
            with torch.no_grad():
                move_scores = model_p2(inputs)
            predicted_move = torch.argmax(move_scores).item()

        print(f"Player {current_player} selects move {predicted_move}")
        game.make_move(predicted_move)
        print(game.get_board_state())

    p1_score, p2_score = game.get_score()
    print("\nGame Over!")
    print(f"Final scores - Player 1: {p1_score}, Player 2: {p2_score}")
    if p1_score > p2_score:
        print("Player 1 wins!")
    elif p2_score > p1_score:
        print("Player 2 wins!")
    else:
        print("It's a tie!")

# Play the game
play_game()
