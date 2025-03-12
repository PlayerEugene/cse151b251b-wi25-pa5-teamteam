from engine import MancalaGame

def print_board(board):
    """Print the mancala board in a more intuitive ASCII art format"""
    print("\n    <------- Player 2's Side -------")
    print("    ", end="")
    for i in range(12, 6, -1):
        print(f"+-----", end="")
    print("+")
    
    # Player 2's pockets with indices on top
    print(f"     |", end="")
    for i in range(12, 6, -1):
        print(f" [{i:2d}]|", end="")
    print()
    
    # Player 2's store and pocket values
    print(f" P2  |", end="")
    for i in range(12, 6, -1):
        print(f"  {board[i]:2d} |", end="")
    print(f" P1")
    
    print(f"({board[13]:2d}) ", end="")
    for i in range(12, 6, -1):
        print(f"+-----", end="")
    print(f"+ ({board[6]:2d})")
    
    # Player 1's pocket values and indices
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

def play_game():
    game = MancalaGame()
    
    while True:
        print_board(game.get_board_state())
        current_player = game.get_current_player()
        valid_moves = game.get_valid_moves()
        
        print(f"Player {current_player}'s turn")
        print(f"Valid moves: {valid_moves}")
        
        while True:
            try:
                move = int(input("Enter pocket number: "))
                if game.make_move(move):
                    break
                else:
                    print("Invalid move, try again")
            except ValueError:
                print("Please enter a valid number")
        
        if game.is_game_over():
            break
    
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
    play_game()
