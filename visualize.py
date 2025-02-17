
def print_board(state: list[int]) -> None:
    """
    Helper function to print the board in a readable format.
    
    Args:
        state: A list of 16 integers representing the current board state.
    """
    print("+----+----+----+----+")
    for i in range(4):
        row = state[i*4:(i+1)*4]
        print("|", ' | '.join('  ' if x == 0 else f'{x:2}' for x in row), "|")
        print("+----+----+----+----+")


def print_solution(path: list[list[int]]) -> None:
    """
    Prints the solution path in a readable format.
    
    Args:
        path: A list of board states representing the solution path, where each state
            is a list of 16 integers representing the board configuration.
            
    Returns:
        None
    """
    if path is None:
        print("No solution found")
        return
        
    print(f"Solution found with {len(path)-1} moves:")
    for i, state in enumerate(path):
        print(f"\nStep {i}:")
        print_board(state)
        