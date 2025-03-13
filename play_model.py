import random
import torch
from MancalaModel import MancalaModel
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

def load_model():
    model = MancalaModel()
    model.load_state_dict(torch.load('mancala_model.pth'))
    model.eval()
    
    return model

def play_model():
    model = load_model()

    user_player = input("Which player would you like to control? (1 or 2): ")
    while user_player not in ['1', '2']:
        user_player = input("Invalid input. Please enter 1 or 2: ")
    user_player = int(user_player)

    game = MancalaGame()

    while not game.is_game_over():
        print_board(game.get_board_state())
        current_player = game.get_current_player()
        valid_moves = game.get_valid_moves()
        
        print(f"Player {current_player}'s turn")
        print(f"Valid moves: {valid_moves}")
        
        if current_player == user_player:
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
        
        else:
            with torch.no_grad():
                inputs = torch.tensor((game.board[0:6] + game.board[7:13] + [current_player]), dtype=torch.float32)
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
                predicted_move = torch.argmax(move_scores).item()
                if predicted_move >= 6:
                    predicted_move += 1
                
                print(f"Player {current_player} (Model) plays: {predicted_move}")
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
