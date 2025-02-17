from heapq import heappush, heappop
from collections import deque
from itertools import combinations
from visualize import print_board, print_solution


def count_inversions(state: list[int]) -> int:
    """Count inversions in the puzzle state (excluding the blank/0).
    
    Args:
        state: A list of integers representing the puzzle state, where 0 represents
              the blank tile.
    
    Returns:
        The number of inversions in the puzzle state, excluding the blank tile.
        An inversion occurs when a tile precedes another tile with a lower number.
    """
    tiles = [x for x in state if x != 0]  # Exclude blank tile
    inversions = 0
    for i in range(len(tiles)):
        for j in range(i + 1, len(tiles)):
            if tiles[i] > tiles[j]:
                inversions += 1
    return inversions


def is_solvable(state: list[int]) -> bool:
    """Check if the puzzle state is solvable.
    
    Args:
        state: A list of integers representing the puzzle state, where 0 represents
              the blank tile.
    
    Returns:
        bool: True if the puzzle state is solvable, False otherwise. A puzzle state
        is solvable if the parity of the blank tile's row from bottom matches the
        parity of inversions count.
    """
    # Verify input has 16 tiles (4x4 grid)
    if len(state) != 16:
        raise ValueError("Input state must contain exactly 16 tiles for a 4x4 grid")
    
    # Verify tiles are in range 0-15 and each appears exactly once
    tile_set = set(state)
    if len(tile_set) != 16 or not all(0 <= x <= 15 for x in state):
        raise ValueError("Input state must contain each number from 0-15 exactly once")
    
    grid_width = 4  # For a 15-puzzle (4x4 grid)
    blank_position = state.index(0)
    blank_row_from_bottom = (grid_width - (blank_position // grid_width))
    inversions = count_inversions(state)
    
    # For even grid width:
    # - Solvable if (inversions + blank_row_from_bottom) is odd
    return (inversions + blank_row_from_bottom) % 2 == 1


def find_legal_moves(state: list[int]) -> list[list[int]]:
    """
    Given a state of the 15-puzzle, returns a list of possible next states.
    
    Args:
        state: A list of 16 integers representing the current board state,
              where 0 represents the empty square. The integers should be
              in the range 0-15.
    
    Returns:
        A list of possible next states, where each state is a list of 16 integers
        representing a valid board configuration reachable in one move.
    """
    # Find the empty square (0)
    empty_pos = state.index(0)
    
    # Calculate row and column of empty square
    row = empty_pos // 4
    col = empty_pos % 4
    
    # List to store possible moves
    possible_moves = []
    
    # Check all possible moves (up, down, left, right)
    # For each move, we'll create a new state by swapping the empty square
    # with the adjacent number
    
    # Move up
    if row > 0:
        new_state = state.copy()
        swap_pos = empty_pos - 4
        new_state[empty_pos], new_state[swap_pos] = new_state[swap_pos], new_state[empty_pos]
        possible_moves.append(new_state)
    
    # Move down
    if row < 3:
        new_state = state.copy()
        swap_pos = empty_pos + 4
        new_state[empty_pos], new_state[swap_pos] = new_state[swap_pos], new_state[empty_pos]
        possible_moves.append(new_state)
    
    # Move left
    if col > 0:
        new_state = state.copy()
        swap_pos = empty_pos - 1
        new_state[empty_pos], new_state[swap_pos] = new_state[swap_pos], new_state[empty_pos]
        possible_moves.append(new_state)
    
    # Move right
    if col < 3:
        new_state = state.copy()
        swap_pos = empty_pos + 1
        new_state[empty_pos], new_state[swap_pos] = new_state[swap_pos], new_state[empty_pos]
        possible_moves.append(new_state)
    
    return possible_moves


def bfs_search(initial_state: list[int], goal_state: list[int], max_states: int = 10000000) -> tuple[list[list[int]] | None, int]:
    """
    Performs breadth-first search to find the solution path from initial_state to goal_state.
    
    Args:
        initial_state: Starting configuration of the puzzle as a list of 16 integers.
        goal_state: Target configuration to reach as a list of 16 integers.
        max_states: Maximum number of states to explore before giving up.
            Defaults to 10000000.

    Returns:
        A tuple containing:
            - The solution path as a list of states (each state being a list of 16 integers),
              or None if no solution is found
            - The number of states explored during the search
    """
    
    # Convert states to tuples for hashable set membership
    initial_state = tuple(initial_state)
    goal_state = tuple(goal_state)
    
    # Queue of (state, path) pairs
    queue = deque([(initial_state, [initial_state])])
    
    # Set of visited states
    visited = {initial_state}
    
    # Counter for explored states
    states_explored = 0
    
    while queue and states_explored < max_states:
        current_state, path = queue.popleft()
        states_explored += 1
        
        if current_state == goal_state:
            return path, states_explored
            
        # Generate and explore all possible next states
        for next_state in find_legal_moves(list(current_state)):
            next_state = tuple(next_state)
            if next_state not in visited:
                visited.add(next_state)
                queue.append((next_state, path + [next_state]))
    
    return None, states_explored


def dfs_search(initial_state: list[int], goal_state: list[int], max_depth: int = 50, max_states: int = 10000000):
    """
    Performs depth-first search to find the solution path from initial_state to goal_state.
    
    Args:
        initial_state: Starting configuration of the puzzle as a list of 16 integers.
        goal_state: Target configuration to reach as a list of 16 integers.
        max_depth: Maximum depth to search to prevent infinite paths.
            Defaults to 50.
        max_states: Maximum number of states to explore before giving up.
            Defaults to 10000000.
        
    Returns:
        A tuple containing:
            - The solution path as a list of states (each state being a list of 16 integers),
              or None if no solution is found
            - The number of states explored during the search
    """
    initial_state = tuple(initial_state)
    goal_state = tuple(goal_state)
    
    # Stack now stores (state, path)
    stack = [(initial_state, [initial_state])]
    visited = {initial_state}
    states_explored = 0
    
    while stack and states_explored < max_states:
        current_state, path = stack.pop()
        current_depth = len(path) - 1  # Calculate depth from path length
        states_explored += 1
        
        if current_state == goal_state:
            return path, states_explored
            
        if current_depth < max_depth:
            # Get all possible next moves
            for next_state in find_legal_moves(list(current_state)):
                next_state = tuple(next_state)
                if next_state not in visited:
                    visited.add(next_state)
                    stack.append((next_state, path + [next_state]))

    return None, states_explored

def calculate_manhattan_distance(state: tuple[int, ...], goal_state: tuple[int, ...]) -> int:
    """
    Calculate the sum of Manhattan distances of all tiles from their goal positions.

    Args:
        state (tuple[int, ...]): Current state of the puzzle as a tuple of integers
        goal_state (tuple[int, ...]): Target state to reach as a tuple of integers

    Returns:
        int: Total Manhattan distance plus linear conflict penalties
    """
    size = 4
    
    # Calculate Manhattan distance
    return sum(
        abs(i//size - goal_state.index(tile)//size) + 
        abs(i%size - goal_state.index(tile)%size)
        for i, tile in enumerate(state) if tile != 0
    )

def calculate_manhattan_distance_with_linear_conflicts(state: tuple[int, ...], goal_state: tuple[int, ...]) -> int:
    """
    Calculate the sum of Manhattan distances of all tiles from their goal positions,
    plus a linear penalty when tiles in same row/column are in wrong order.

    Args:
        state (tuple[int, ...]): Current state of the puzzle as a tuple of integers
        goal_state (tuple[int, ...]): Target state to reach as a tuple of integers

    Returns:
        int: Total Manhattan distance plus linear conflict penalties
    """
    size = 4
    
    # Calculate Manhattan distance
    manhattan = calculate_manhattan_distance(state, goal_state)
    
    # Helper to find conflicts in a line (row or column)
    def count_conflicts(tiles):
        return 2 * sum(1 for (p1, t1), (p2, t2) in combinations(tiles, 2)
                      if p1 < p2 and t1 > t2)
    
    # Row conflicts
    row_conflicts = sum(
        count_conflicts([
            (col, state[row*size + col]) 
            for col in range(size)
            if state[row*size + col] != 0 
            and goal_state.index(state[row*size + col])//size == row
        ])
        for row in range(size)
    )
    
    # Column conflicts
    col_conflicts = sum(
        count_conflicts([
            (row, state[row*size + col])
            for row in range(size)
            if state[row*size + col] != 0
            and goal_state.index(state[row*size + col])%size == col
        ])
        for col in range(size)
    )
    
    return manhattan + row_conflicts + col_conflicts

def astar_search(initial_state: list[int], goal_state: list[int], max_states: int = 10000000):
    """
    A* search using Manhattan distance heuristic.

    Args:
        initial_state (list[int]): The starting configuration of the puzzle
        goal_state (list[int]): The target configuration to reach
        max_states (int, optional): Maximum number of states to explore. Defaults to 10000000.

    Returns:
        tuple: A tuple containing:
            - list: The solution path from initial to goal state, or None if no solution found
            - int: Number of states explored during search
    """
    initial_state = tuple(initial_state)
    goal_state = tuple(goal_state)
    
    # Priority queue of (f_score, g_score, state, path)
    # f_score = g_score + h_score (path cost + heuristic)
    queue = [(0 + calculate_manhattan_distance(initial_state, goal_state), 0, initial_state, [initial_state])]
    
    # Keep track of best g_score for each state
    g_scores = {initial_state: 0}
    states_explored = 0
    
    while queue and states_explored < max_states:
        f_score, g_score, current_state, path = heappop(queue)
        h_score = f_score - g_score
        states_explored += 1
        
        if current_state == goal_state:
            return path, states_explored
            
        # Generate successor states
        for next_state in find_legal_moves(list(current_state)):
            next_state = tuple(next_state)
            new_g_score = g_score + 1  # Cost of 1 to move to any adjacent state
            
            # If we haven't seen this state or found a better path to it
            if next_state not in g_scores or new_g_score < g_scores[next_state]:
                g_scores[next_state] = new_g_score
                h_score = calculate_manhattan_distance_with_linear_conflicts(next_state, goal_state)
                f_score = new_g_score + h_score
                heappush(queue, (f_score, new_g_score, next_state, path + [next_state]))
    
    return None, states_explored


def reconstruct_path(forward_path: list[list[int]], backward_path: list[list[int]]) -> list[list[int]]:
    """Reconstruct the final solution path by combining forward and backward paths.
    
    Args:
        forward_path: List of states from initial state to meeting point
        backward_path: List of states from goal state to meeting point
        
    Returns:
        list[list[int]]: Combined path from initial state to goal state, with
            duplicate meeting point state removed
    """
    # Remove the duplicate state at the meeting point
    return forward_path[:-1] + backward_path[::-1]


def bidirectional_astar(initial_state: list[int], goal_state: list[int], max_states: int = 10000000):
    """
    Bidirectional A* search that searches simultaneously from initial and goal states. 
    Note that this is a simplified implementation that accepts the first solution found, 
    which may not be the shortest path.

    Args:
        initial_state: Starting configuration of the puzzle
        goal_state: Target configuration to reach
        max_states: Maximum number of states to explore before giving up
        
    Returns:
        tuple: Contains:
            - The solution path from initial to goal state, or None if no solution found
            - Number of states explored during search
    """
    # Convert states to tuples for hashing
    initial_state = tuple(initial_state)
    goal_state = tuple(goal_state)
    
    # Forward search
    forward_queue = [(calculate_manhattan_distance_with_linear_conflicts(initial_state, goal_state), 0, initial_state, [initial_state])]
    forward_g_scores = {initial_state: 0}

    # Backward search
    backward_queue = [(calculate_manhattan_distance_with_linear_conflicts(goal_state, initial_state), 0, goal_state, [goal_state])]
    backward_g_scores = {goal_state: 0}

    states_explored = 0
    visited_forward = {}
    visited_backward = {}

    while forward_queue and backward_queue and states_explored < max_states:
        # Expand the smaller queue
        if len(forward_queue) <= len(backward_queue):
            _, g_score, current_state, path = heappop(forward_queue)
            visited_forward[current_state] = path
            states_explored += 1

            # Check for meeting point
            if current_state in visited_backward:
                return reconstruct_path(path, visited_backward[current_state]), states_explored

            # Expand neighbors
            for neighbor in find_legal_moves(list(current_state)):
                tentative_g_score = g_score + 1
                neighbor = tuple(neighbor)
                if neighbor not in forward_g_scores or tentative_g_score < forward_g_scores[neighbor]:
                    forward_g_scores[neighbor] = tentative_g_score
                    f_score = tentative_g_score + calculate_manhattan_distance_with_linear_conflicts(neighbor, goal_state)
                    heappush(forward_queue, (f_score, tentative_g_score, neighbor, path + [neighbor]))
        else:
            _, g_score, current_state, path = heappop(backward_queue)
            visited_backward[current_state] = path
            states_explored += 1

            # Check for meeting point
            if current_state in visited_forward:
                return reconstruct_path(visited_forward[current_state], path), states_explored

            # Expand neighbors
            for neighbor in find_legal_moves(list(current_state)):
                tentative_g_score = g_score + 1
                neighbor = tuple(neighbor)
                if neighbor not in backward_g_scores or tentative_g_score < backward_g_scores[neighbor]:
                    backward_g_scores[neighbor] = tentative_g_score
                    f_score = tentative_g_score + calculate_manhattan_distance_with_linear_conflicts(neighbor, initial_state)
                    heappush(backward_queue, (f_score, tentative_g_score, neighbor, path + [neighbor]))

    return None, states_explored  # No solution found within max_states


if __name__ == "__main__":

    initial_state = [1, 3, 2, 0,
                    5, 6, 4, 8,
                    9, 10, 7, 11,
                    13, 14, 15, 12]

    goal_state = [1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 0]
    
    print(f"Puzzle is solvable: {is_solvable(initial_state)}")

    max_states = 10000000
    max_depth = 50

    print(f"Running BFS search with a maximum of {max_states} states ...")
    path, states_explored = bfs_search(initial_state, goal_state, max_states)
    print_solution(path)
    print(f"States explored: {states_explored}")
    print()

    print(f"Running DFS search with a maximum of {max_states} states and a maximum depth of {max_depth} ...")
    path, states_explored = dfs_search(initial_state, goal_state, max_depth, max_states)
    print_solution(path)
    print(f"States explored: {states_explored}")
    print()

    print(f"Running A* search with a maximum of {max_states} states ...")
    path, states_explored = astar_search(initial_state, goal_state, max_states)
    print_solution(path)
    print(f"States explored: {states_explored}")
    print()

    print(f"Running Bidirectional A* search with a maximum of {max_states} states ...")
    path, states_explored = bidirectional_astar(initial_state, goal_state, max_states)
    print_solution(path)
    print(f"States explored: {states_explored}")
    print()


