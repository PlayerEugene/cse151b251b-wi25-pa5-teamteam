import random
import torch
from MancalaCNN import MancalaCNN
from engine import MancalaGame

def print_board(board):
    print("\n    <------- Player 2's Side -------")
    print("    ", end="")
    for i in range(12, 6, -1):
        print(f"+-----", end="")
    print("+")
    
    print(f"     |", end="")
    for i in range(12, 6, -1):
        print(f" [{i:2d}]|", end="")
    print()
    
    print(f" P2  |", end="")
    for i in range(12, 6, -1):
        print(f"  {board[i]:2d} |", end="")
    print(f" P1")
    
    print(f"({board[13]:2d}) ", end="")
    for i in range(12, 6, -1):
        print(f"+-----", end="")
    print(f"+ ({board[6]:2d})")
    
    print(f"     |", end="")
    for i in range(6):
        print(f"  {board[i]:2d} |", end="")
    print()
    
    print(f"     |", end="")
    for i in range(6):
        print(f" [{i:2d}]|", end="")
    print()
    
    print("    ", end="")
    for i in range(6):
        print(f"+-----", end="")
    print("+")
    print("    ------> Player 1's Side --------\n")

def load_model():
    model = MancalaCNN()
    model.load_state_dict(torch.load('mancala_model.pth'))
    model.eval()
    
    return model

def play_random():
    model = load_model()
    
    game = MancalaGame()

    random_side = random.randint(1, 2)
    
    while not game.is_game_over():
        current_player = game.get_current_player()
        valid_moves = game.get_valid_moves()
        
        if current_player == random_side:
            move = random.choice(valid_moves)
            game.make_move(move)
        
        else:
            with torch.no_grad():
                p1_side = torch.tensor(game.board[0:6], dtype=torch.float32).unsqueeze(0)
                p2_side = torch.tensor(game.board[7:13], dtype=torch.float32).unsqueeze(0)
                
                player_turn = torch.tensor([1.0 if current_player == 1 else -1.0] * 6, dtype=torch.float32).unsqueeze(0)

                inputs = torch.cat((p1_side, p2_side, player_turn), dim=1)
                inputs = inputs.view(1, 3, 1, 6)
                
                move_scores, state_value = model(inputs)

                for move in range(move_scores.shape[-1]):
                    if current_player == 1:
                        if move not in valid_moves:
                            move_scores[0, move] = float('-inf')
                    else:
                        if move + 1 not in valid_moves:
                            move_scores[0, move] = float('-inf')
                predicted_move = torch.argmax(move_scores).item()
                if predicted_move >= 6:
                    predicted_move += 1

                game.make_move(predicted_move)
    
    p1_score, p2_score = game.get_score()
    
    if p1_score > p2_score:
        return 1, random_side
    elif p2_score > p1_score:
        return 2, random_side
    else:
        return 0, random_side

# Function to simulate 100 games and track results
def simulate_games(num_games=100):
    random_wins = 0
    model_wins = 0
    draws = 0
    
    for _ in range(num_games):
        result, random_side = play_random()
        if result == random_side:
            random_wins += 1
        elif result == 0:
            draws += 1
        else:
            model_wins += 1
        print(f"Game {_} done")
    
    print(f"After {num_games} games:")
    print(f"Random wins: {random_wins}")
    print(f"Model wins: {model_wins}")
    print(f"Draws: {draws}")

if __name__ == "__main__":
    simulate_games(num_games=1000)
