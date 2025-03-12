class MancalaGame:
    def __init__(self):
        # Player 1's pockets are 0-5, store at 6
        # Player 2's pockets are 7-12, store at 13
        self.board = [4] * 14
        self.board[6] = 0  # Player 1's store
        self.board[13] = 0  # Player 2's store
        self.current_player = 1

    def is_game_over(self):
        # game ends if either player's side is empty
        p1_empty = all(self.board[i] == 0 for i in range(6))
        p2_empty = all(self.board[i] == 0 for i in range(7, 13))
        
        # If game is over, move all remaining stones to the appropriate store
        if p1_empty or p2_empty:
            if p1_empty:
                for i in range(7, 13):
                    self.board[13] += self.board[i]
                    self.board[i] = 0
            if p2_empty:
                for i in range(6):
                    self.board[6] += self.board[i]
                    self.board[i] = 0
                
        return p1_empty or p2_empty

    def make_move(self, pocket):
        """
        Makes a move from the selected pocket
        Returns True if move was valid, False otherwise
        """
        # Validate move
        if self.current_player == 1 and not 0 <= pocket <= 5:
            return False
        if self.current_player == 2 and not 7 <= pocket <= 12:
            return False
        if self.board[pocket] == 0:
            return False

        stones = self.board[pocket]
        self.board[pocket] = 0
        current_pocket = pocket

        extra_turn = False
        while stones > 0:
            current_pocket = (current_pocket + 1) % 14
            if (self.current_player == 1 and current_pocket == 13) or (self.current_player == 2 and current_pocket == 6):
                continue
            
            self.board[current_pocket] += 1
            stones -= 1

            # Capture rule: If last stone lands in empty pocket on player's side
            if stones == 0:
                if self.current_player == 1 and 0 <= current_pocket <= 5:
                    opposite = 12 - current_pocket  # Maps 0->12, 1->11, ..., 5->7
                    if self.board[current_pocket] == 1 and self.board[opposite] > 0:
                        # capture stones
                        self.board[6] += self.board[opposite] + 1 
                        self.board[opposite] = 0  # empty opposite pocket
                        self.board[current_pocket] = 0  # empty current pocket
                
                elif self.current_player == 2 and 7 <= current_pocket <= 12:
                    opposite = 12 - current_pocket  # Maps 12->0, 11->1, ..., 7->5
                    if self.board[current_pocket] == 1 and self.board[opposite] > 0:
                        self.board[13] += self.board[opposite] + 1  
                        self.board[opposite] = 0 
                        self.board[current_pocket] = 0

                if (self.current_player == 1 and current_pocket == 6) or (self.current_player == 2 and current_pocket == 13):
                    extra_turn = True

        if self.is_game_over():
            return True
        if not extra_turn:
            self.current_player = 3 - self.current_player

        return True

    def get_valid_moves(self):
        """Returns list of valid moves for current player"""
        if self.current_player == 1:
            return [i for i in range(6) if self.board[i] > 0]
        else:
            return [i for i in range(7, 13) if self.board[i] > 0]

    def get_score(self):
        """Returns (player1_score, player2_score)"""
        return self.board[6], self.board[13]

    def get_board_state(self):
        """Returns current board state"""
        return self.board.copy()

    def get_current_player(self):
        """Returns current player (1 or 2)"""
        return self.current_player
