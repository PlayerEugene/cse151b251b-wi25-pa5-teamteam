import random
import torch
from MancalaModel import MancalaModelMCTS
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

def play_model():
    mcts_agent = MancalaModelMCTS(num_simulations=500, ucb_c=1.4)

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
            best_move = mcts_agent.mcts(game)
            game.make_move(best_move)
                
            print(f"Player {current_player} (Model) plays: {best_move}")
        
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
