import numpy as np
from numba import njit
from .performance_metrics import calc_TTS
from .max3cut_utils import read_graph_file, load_optimal_solution, count_wrong_edges

def get_interaction_values_HO(edges, num_vertices, num_spins, B, A=1.):
    """
    Order-2 and order-4 interaction values for the higher-order Ising formulation
    of the Max-3-Cut problem as described in Eq. (5) of https://arxiv.org/pdf/2508.00565v2.
    Parameters
    ----------
    edges : np.ndarray
        Array of edges in the graph.
    num_vertices : int
        Number of vertices in the graph.
    num_spins : int
        Total number of spins in the system (= 3 * num_vertices).
    B : float
        Hyperparameter of the higher-order Ising formulation. (cf. Eq. (5) in https://arxiv.org/pdf/2508.00565v2).
    A : float, optional
        Hyperparameter of the higher-order Ising formulation. Default is 1.
    """
    # quadratic interactions
    J_order2 = np.zeros((num_spins,num_spins),dtype=np.float64)
    for i in range(num_spins):
        for j in range(num_spins):
            v1, c1 = i % num_vertices, i // num_vertices
            v2, c2 = j % num_vertices, j // num_vertices
            if v1==v2 and c1!=c2:
                J_order2[i,j] -= A
    # fourth-order interactions
    o4_ids = []
    o4_items = []
    for u,v in edges:
        for j in range(3):
            for i in range(j):
                    ids = [i*num_vertices+u,i*num_vertices+v,j*num_vertices+u,j*num_vertices+v]
                    assert len(np.unique(ids)) == 4
                    o4_ids.append(tuple(sorted(ids)))
                    o4_items.append(-B)
    return J_order2, np.array(o4_ids, dtype=np.int64), np.array(o4_items, dtype=np.float64)

@njit
def amps_to_states_HO(X,num_vertices):
    """
    Convert each spin triplet to an integer Potts state (i.e., a color) 
    according to Eq. (6) in https://arxiv.org/pdf/2508.00565v2.
    Mapping:
        0 → 'red'
        1 → 'green'
        2 → 'blue'
       -1 → 'undefined color'
    """
    states = np.zeros(num_vertices,dtype=np.int64)
    for i in range(num_vertices):
        triplet = X[i:len(X):num_vertices]
        if len(triplet) != 3:
            raise ValueError(f"Expected triplet of length 3, got {len(triplet)} for vertex {i}")

        num_up = np.sum(triplet > 0)
        if num_up == 0 or num_up == 3: # all up or all down → undefined color
            states[i] = -1
        elif num_up == 1: # one-hot encoding
            states[i] = np.argmax(triplet)
        elif num_up == 2: # inverted one-hot encoding
            states[i] = np.argmin(triplet)
        else:
            raise RuntimeError(f"Unexpected number of positive entries: {num_up}")
        
    return states

@njit
def update_spins_HO(X, alpha, beta, J_o2, o4_ids, o24_vals, dt, gamma=0.001):
    """
    Update spins X using Euler integration for the higher-order Ising formulation
    according to Eqs. (A1) and (9) in https://arxiv.org/pdf/2508.00565v2.
    Parameters
    ----------
    X : np.ndarray
        Current spin amplitudes.
    alpha : float
        Linear gain.
    beta : float
        Interaction strength.
    J_o2 : np.ndarray
        Coupling matrix for order-2 interactions.
    o4_ids : np.ndarray
        Indices for order-4 interactions.
    o24_vals : np.ndarray
        Values for order-4 interactions.
    dt : float
        Time step for Euler integration.
    gamma : float, optional
        Stochastic noise strength.
    Returns
    -------
    np.ndarray
        Updated spin amplitudes.
    """
    sign_X = np.sign(X)
    terms_o4 = np.zeros(len(X), dtype=np.float64)
    for i in range(len(o24_vals)):
        idx0, idx1, idx2, idx3 = o4_ids[i]
        item = o24_vals[i]
        s0, s1, s2, s3 = sign_X[idx0], sign_X[idx1], sign_X[idx2], sign_X[idx3]
        terms_o4[idx0] += item * s1 * s2 * s3
        terms_o4[idx1] += item * s0 * s2 * s3
        terms_o4[idx2] += item * s0 * s1 * s3
        terms_o4[idx3] += item * s0 * s1 * s2
    dXdt = -X + np.tanh(alpha * X + beta * (np.dot(J_o2,sign_X) + terms_o4))
    X += dXdt * dt
    if gamma != 0:
        X += np.random.normal(0,np.sqrt(dt),size=len(X)) * gamma
    return X

@njit
def euler_integration_HO(num_inits, alpha, annealing_speed, J_o2, o4_ids, o4_items, dt, max_time, optimal_solution, num_vertices, num_spins, edges):
    """
    Perform Euler integration of the higher-order Ising model dynamics
    for multiple independent initializations.
    Parameters
    ----------
    num_inits : int
        Number of independent initializations (runs).
    alpha : float
        Linear gain.
    annealing_speed : float
        Rate of change of interaction strength (d(beta)/dt).
    J_o2 : np.ndarray
        Coupling matrix for order-2 interactions.
    o4_ids : np.ndarray
        Indices for order-4 interactions.
    o4_items : np.ndarray
        Values for order-4 interactions.
    dt : float
        Time step for Euler integration.
    max_time : float
        Maximum time for Euler integration.
    optimal_solution : int
        Optimal Max-3-Cut solution value (minimal number of wrong edges).
    num_spins : int
        Total number of spins in the system.
    edges : np.ndarray
        Array of edges in the graph.
    Returns
    -------
    number_of_successes : float
        Number of successful runs that found the optimal solution.
    times_success : np.ndarray
        Array of times at which the optimal solution was found for each initialization.
    """
    beta_range = annealing_speed * np.arange(0, max_time+dt, dt)
    number_of_successes = 0.
    times_success = np.zeros(num_inits, dtype=np.float64)-1.
    for idx_init in range(num_inits):
        X = (np.random.rand(num_spins)*2-1)*10**(-10) # random small init
        for idx_beta, beta in enumerate(beta_range):
            X = update_spins_HO(X, alpha, beta, J_o2, o4_ids, o4_items, dt)

            # calculate binary energy
            states = amps_to_states_HO(X,num_vertices)
            num_wrong_edges = count_wrong_edges(states,edges)
            
            if num_wrong_edges == optimal_solution:
                number_of_successes += 1
                times_success[idx_init] = (idx_beta+1)*dt
                break
            else:
                assert int(num_wrong_edges) > int(optimal_solution)

    return number_of_successes, times_success

def run_HO(problem_name, annealing_speed, alpha, B, dir_graphs, dir_solutions, num_inits=100, max_time=10000., dt=0.01):
    """
    Run the higher-order Ising formulation simulation for a given Max-3-Cut problem instance.
    Parameters
    ----------
    problem_name : str
        Name of the graph file.
    annealing_speed : float
        Rate of change of interaction strength (d(beta)/dt). (cf. Eq. (A1) in https://arxiv.org/pdf/2508.00565v2)
    alpha : float
        Linear gain. (cf. Eq. (A1) in https://arxiv.org/pdf/2508.00565v2)
    B : float
        Hyperparameter of the higher-order Ising formulation. (cf. Eq. (5) in https://arxiv.org/pdf/2508.00565v2). A=1 by default.
    dir_graphs : str, optional
        Directory containing the graph files.
    dir_solutions : str, optional
        Directory containing the optimal solutions CSV file.
    num_inits : int, optional
        Number of independent initializations (runs).
    max_time : float, optional
        Maximum time for Euler integration.
    dt : float, optional
        Time step for Euler integration.
    Returns
    -------
    SR : float
        Success rate of finding the optimal solution.
    TTS : float
        Time-to-solution for the problem instance.
    """
    # load problem instance 
    edges, _, num_vertices = read_graph_file(problem_name, dir_graphs)
    optimal_solution = load_optimal_solution(problem_name, dir_solutions)

    # get interaction values for higher-order Ising formulation
    num_spins = num_vertices * 3 # each vertex is represented by 3 spins
    J_o2, o4_ids, o4_items = get_interaction_values_HO(edges, num_vertices, num_spins, B)

    # perform Euler integration
    number_of_successes, times_success = euler_integration_HO(num_inits, alpha, annealing_speed, J_o2, o4_ids, o4_items, dt, max_time, optimal_solution, num_vertices, num_spins, edges)

    # calculate SR and TTS
    SR = number_of_successes/num_inits
    TTS = calc_TTS(times_success, dt, max_time, num_inits)
    return SR, TTS