import random
import torch
from MancalaCNN import MancalaCNN
from engine import MancalaGame

# Function to print the board in an intuitive ASCII art format
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

# Function to load only Player 2's model (since Player 1 is controlled by the human)
def load_model():
    model_p2 = MancalaCNN()
    
    # Load pre-trained model weights for Player 2
    model_p2.load_state_dict(torch.load('mancala_model_p2.pth'))
    
    # Set model to evaluation mode (disables dropout, etc.)
    model_p2.eval()
    
    return model_p2

# Function to let the model play against the user (human)
def play_model():
    # Load model for Player 2 (the trained model)
    model_p2 = load_model()
    
    game = MancalaGame()
    
    while not game.is_game_over():
        print_board(game.get_board_state())
        current_player = game.get_current_player()
        valid_moves = game.get_valid_moves()
        
        print(f"Player {current_player}'s turn")
        print(f"Valid moves: {valid_moves}")
        
        if current_player == 1:  # Human plays Player 1
            while True:
                try:
                    move = int(input("Enter pocket number: "))
                    if move in valid_moves:
                        if game.make_move(move):
                            break
                        else:
                            print("Invalid move, try again")
                    else:
                        print("Invalid move, try again")
                except ValueError:
                    print("Please enter a valid number")
        
        else:  # Model plays Player 2
            with torch.no_grad():
                # Get board state for Player 2
                p1_side = torch.tensor(game.board[0:6], dtype=torch.float32).unsqueeze(0)
                p2_side = torch.tensor(game.board[7:13], dtype=torch.float32).unsqueeze(0)
                
                inputs = torch.stack((p1_side, p2_side), dim=0).unsqueeze(0)
                
                move_scores = model_p2(inputs)
                # Mask invalid moves
                for move in range(move_scores.shape[-1]):
                    if move + 7 not in valid_moves:
                        move_scores[0, move] = float('-inf')  # Mask invalid moves
                predicted_move = torch.argmax(move_scores).item() + 7
                
                print(f"Player 2 (Model) plays: {predicted_move}")
                game.make_move(predicted_move)
        
    print("\nGame Over!")
    print_board(game.get_board_state())
    p1_score, p2_score = game.get_score()
    print(f"Final scores - Player 1: {p1_score}, Player 2: {p2_score}")
    
    if p1_score > p2_score:
        print("Player 1 wins!")
    elif p2_score > p1_score:
        print("Player 2 wins!")
    else:
        print("It's a tie!")

if __name__ == "__main__":
    play_model()
