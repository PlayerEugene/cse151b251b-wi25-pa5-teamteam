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
    model_p2 = MancalaCNN()
    model_p2.load_state_dict(torch.load('mancala_model_p2.pth'))
    model_p2.eval()
    
    return model_p2

def play_random():
    model_p2 = load_model()
    
    game = MancalaGame()
    
    while not game.is_game_over():
        current_player = game.get_current_player()
        valid_moves = game.get_valid_moves()
        
        if current_player == 1:
            move = random.choice(valid_moves)
            game.make_move(move)
        
        else:
            with torch.no_grad():
                p1_side = torch.tensor(game.board[0:6], dtype=torch.float32).unsqueeze(0)
                p2_side = torch.tensor(game.board[7:13], dtype=torch.float32).unsqueeze(0)
                
                inputs = torch.stack((p1_side, p2_side), dim=0).unsqueeze(0)
                
                move_scores = model_p2(inputs)

                for move in range(move_scores.shape[-1]):
                    if move + 7 not in valid_moves:
                        move_scores[0, move] = float('-inf')
                predicted_move = torch.argmax(move_scores).item() + 7
                
                game.make_move(predicted_move)
    
    p1_score, p2_score = game.get_score()
    
    if p1_score > p2_score:
        return 1
    elif p2_score > p1_score:
        return 2
    else:
        return 0

# Function to simulate 100 games and track results
def simulate_games(num_games=100):
    wins_p1 = 0
    wins_p2 = 0
    draws = 0
    
    for _ in range(num_games):
        result = play_random()
        if result == 1:
            wins_p1 += 1
        elif result == 2:
            wins_p2 += 1
        else:
            draws += 1
    
    print(f"After {num_games} games:")
    print(f"Player 1 wins: {wins_p1}")
    print(f"Player 2 wins: {wins_p2}")
    print(f"Draws: {draws}")

if __name__ == "__main__":
    simulate_games(num_games=1000)
