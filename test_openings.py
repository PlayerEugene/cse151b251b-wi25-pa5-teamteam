from engine import MancalaGame
from MancalaModel import MancalaModelMCTS
from collections import defaultdict
import json
import os

def analyze_openings(num_simulations=1000, num_moves=3, save_path="opening_stats.json"):
    """
    Analyze Mancala openings using MCTS to determine winning probabilities
    for different opening sequences.
    
    Args:
        num_simulations: Number of MCTS simulations per position
        num_moves: How many moves deep to analyze
        save_path: Path to save the opening statistics
    """
    results = defaultdict(lambda: {'wins': 0, 'total': 0})
    
    mcts_p1 = MancalaModelMCTS(num_simulations=num_simulations, mcts_player=1)
    mcts_p2 = MancalaModelMCTS(num_simulations=num_simulations, mcts_player=2)
    
    def analyze_position(game, depth=0, move_sequence=""):
        if depth >= num_moves * 2:
            return
            
        mcts = mcts_p1 if game.current_player == 1 else mcts_p2
        
        best_move = mcts.mcts(game)
        
        if move_sequence:
            wins = 0
            trials = 100
            for _ in range(trials):
                sim_game = MancalaGame()
                sim_game.board = game.board.copy()
                sim_game.current_player = game.current_player
                
                while not sim_game.is_game_over():
                    curr_mcts = mcts_p1 if sim_game.current_player == 1 else mcts_p2
                    move = curr_mcts.mcts(sim_game)
                    sim_game.make_move(move)
                
                p1_score, p2_score = sim_game.get_score()
                if p1_score > p2_score:
                    wins += 1
                elif p1_score == p2_score:
                    wins += 0.5
                    
            results[move_sequence]['wins'] += wins
            results[move_sequence]['total'] += trials
        
        for move in game.get_valid_moves():
            new_game = MancalaGame()
            new_game.board = game.board.copy()
            new_game.current_player = game.current_player
            new_game.make_move(move)
            new_sequence = f"{move_sequence} {move}" if move_sequence else str(move)
            analyze_position(new_game, depth + 1, new_sequence)

    # Start the analysis from initial position
    initial_game = MancalaGame()
    analyze_position(initial_game)
    
    # Convert defaultdict to regular dict for JSON serialization
    results_dict = {k: dict(v) for k, v in results.items()}
    
    # Save results to JSON file
    with open(save_path, 'w') as f:
        json.dump(results_dict, f, indent=4)
    print(f"\nSaved opening statistics to {save_path}")
    
    # Print results
    print("\nTop Opening Sequences (from Player 1's perspective):")
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1]['wins'] / x[1]['total'],
        reverse=True
    )[:10]
    
    for sequence, stats in sorted_results:
        win_rate = stats['wins'] / stats['total']
        moves = sequence.strip().split()
        print(f"\nMove sequence: {' → '.join(moves)}")
        print(f"Player 1 win rate: {win_rate:.2%}")
        print(f"Based on {stats['total']} simulated games")

if __name__ == "__main__":
    print("Analyzing Mancala openings...")
    analyze_openings(num_simulations=500, num_moves=2, save_path="openings_stats_1000.json")
