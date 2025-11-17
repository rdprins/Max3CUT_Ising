import os
import numpy as np
import pandas as pd
from numba import njit

def read_graph_file(problem_name, directory):
    """
    Read a graph file and return edges, weights, and the number of vertices.

    The file is expected to have the following format:
        First line: num_vertices num_edges
        Subsequent lines: i j weight
        (Vertices are 1-indexed in the file and converted to 0-indexed.)

    Parameters
    ----------
    problem_name : str
        Name of the graph file.
    directory : str
        Directory containing the graph file.

    Returns
    -------
    edges : np.ndarray
        Array of edges as (i, j) pairs.
    weights : list of int
        List of edge weights.
    num_vertices : int
        Number of vertices in the graph.
    """
    # load file
    filepath = os.path.join(directory, problem_name)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Graph file not found: {filepath}")

    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [line.split() for line in lines]
    lines = [[int(x) for x in line] for line in lines]

    if len(lines) == 0:
        raise ValueError("Graph file is empty or malformed.")

    # read first line
    num_vertices, num_edges = lines[0][:2]
    
    # read edges and weights
    edges, weights = [], []
    for line in lines[1:]:
        i, j, val = line
        edges.append((i-1,j-1))
        weights.append(val)

    # consistency checks
    all_vertices = np.array(edges).ravel()
    if num_vertices != len(np.unique(all_vertices)):
        raise ValueError("Number of unique vertices does not match header. There might be isolated vertices.")
    if num_edges != len(edges):
        raise ValueError("Number of edges does not match header.")
    
    # assert that all weights are 1
    if not all(w == 1 for w in weights):
        raise ValueError("All edge weights are expected to be 1.") # extend later if needed

    return np.array(edges,dtype=np.int64), np.array(weights,dtype=np.float64), num_vertices

def load_optimal_solution(problem_name, dir):
    """
    Load the optimal solution for a given problem from a CSV file.
    Parameters
    ----------
    problem_name : str
        Name of the problem instance.
    dir : str
        Directory containing the optimal solutions CSV file.
    Returns
    -------
    int
        Optimal Max-3-Cut solution value (i.e., the minimal number of edges that connect vertices of the same color).
    """
    path_csv = os.path.join(dir, "optimal_solutions.csv")
    if not os.path.exists(path_csv):
        raise FileNotFoundError(f"Optimal solutions file not found: {path_csv}")
    df_sols = pd.read_csv(path_csv)
    sol = df_sols[df_sols['problem_name'] == problem_name]['optimal_solution'].values
    if len(sol) == 0:
        raise ValueError(f"No optimal solution found for problem {problem_name}")
    return int(sol[0])

@njit
def count_wrong_edges(states,edges):
    """
    Count the number of edges that connect vertices of the same color. Undefined colors (state=-1) are also counted as wrong.
    Parameters
    ----------
    states : np.ndarray
        Array of vertex states (values can be 0, 1, 2 for colors, or -1 for undefined).
    edges : np.ndarray
        Array of edges as (i, j) pairs.
    Returns
    -------
    int
        Number of edges connecting vertices of the same color or involving undefined colors.
    """
    num_wrong = 0
    for i,j in edges:
        if (states[i] == states[j]) or (states[i] == -1) or (states[j] == -1):
            num_wrong += 1
    return num_wrong