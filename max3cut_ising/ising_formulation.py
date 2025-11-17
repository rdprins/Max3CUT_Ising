import numpy as np
from numba import njit
from .performance_metrics import calc_TTS
from .max3cut_utils import read_graph_file, load_optimal_solution, count_wrong_edges

def get_interaction_values_quadratic(edges, num_vertices, num_spins, B, A=1., zeta=1.0):
    """
    Order-1 and order-2 interaction values for the quadratic Ising formulations
    of the Max-3-Cut problem as described in https://arxiv.org/pdf/2508.00565v2.
    Eq. (3) of that paper corresponds to zeta=1, while Eq. (7) corresponds to zeta=0.6.

    Parameters
    ----------
    edges : np.ndarray
        Array of edges in the graph.
    num_vertices : int
        Number of vertices in the graph.
    num_spins : int
        Total number of spins (= 3 * num_vertices).
    B : float
        Edge penalty strength.
    A : float, optional
        One-hot penalty strength (default: 1).
    zeta : float, optional
        Rescaling factor for linear terms (default: 1). (cf. Eq. (7) in https://arxiv.org/pdf/2508.00565v2).

    Returns
    -------
    J : np.ndarray
        Quadratic coupling matrix.
    H : np.ndarray
        External field vector.
    """
    J = np.zeros((num_spins,num_spins),dtype=np.float64)
    H = np.zeros(num_spins,dtype=np.float64)
    
    # impose one-hot encoding
    for v in range(num_vertices):
        for i in range(3):
            for j in range(3):
                if i != j:
                    idx0 = i*num_vertices + v
                    idx1 = j*num_vertices + v
                    J[idx0,idx1] += (-A) * 0.25
                    J[idx1,idx0] += (-A) * 0.25
        for i in range(3):
            idx = i*num_vertices + v
            H[idx] = (-A) * 0.5
    
    # max-3-cut constraints (add penalty if edge connects same colors)
    for u,v in edges:
        for i in range(3):
            idx0 = i*num_vertices + u
            idx1 = i*num_vertices + v
            J[idx0,idx1] += (-B) * 0.25
            J[idx1,idx0] += (-B) * 0.25
            H[idx0] += (-B) * 0.25
            H[idx1] += (-B) * 0.25
    
    H *= zeta # rescaling factor
    return J, H

@njit
def amps_to_states_quadratic(X, num_vertices):
    """
    Convert each spin triplet to an integer state using one-hot interpretation.

    Parameters
    ----------
    X : np.ndarray
        Current spin amplitudes. (length = 3 * num_vertices)
    num_vertices : int
        Number of vertices in the system.

    Returns
    -------
    states : np.ndarray
        Color assignments per vertex (0,1,2) or -1 if invalid.
    """
    states = np.zeros(num_vertices,dtype=np.int64)
    for i in range(num_vertices):
        triplet = X[i:len(X):num_vertices] # get the 3 spins of vertex i
        if len(triplet) != 3:
            raise ValueError(f"Expected triplet of length 3, got {len(triplet)} for vertex {i}")
        
        num_up = np.sum(triplet > 0)
        if num_up == 1: # correctly OH-encoded
            states[i] = np.argmax(triplet)
        else: # not correctly OH-encoded
            states[i] = -1
    return states

@njit
def update_spins_quadratic(X, alpha, beta, J, H, dt, gamma=0.001):
    """
    Euler update step for the quadratic Ising formulation.

    Parameters
    ----------
    X : np.ndarray
        Current spin amplitudes.
    alpha : float
        Linear gain.
    beta : float
        Interaction strength.
    J : np.ndarray
        Quadratic coupling matrix.
    H : np.ndarray
        External field vector.
    dt : float
        Time step for Euler integration.
    gamma : float, optional
        Noise strength (default: 0.001).

    Returns
    -------
    np.ndarray
        Updated spin amplitudes.
    """
    sign_X = np.sign(X)
    dXdt = -X + np.tanh(alpha * X + beta * (np.dot(J,sign_X) + H))
    X += dXdt * dt
    if gamma != 0:
        X += np.random.normal(0,np.sqrt(dt),size=len(X)) * gamma
    return X

@njit
def euler_integration_quadratic(num_inits, alpha, annealing_speed, J, H, dt, max_time, optimal_solution, num_vertices, num_spins, edges):
    """
    Perform Euler integration of the quadratic Ising dynamics over multiple initializations.

    Parameters
    ----------
    num_inits : int
        Number of independent runs.
    alpha : float
        Linear gain.
    annealing_speed : float
        Rate of change of beta (d(beta)/dt).
    J : np.ndarray
        Quadratic coupling matrix.
    H : np.ndarray
        External field vector.
    dt : float
        Time step.
    max_time : float
        Maximum integration time.
    optimal_solution : int
        Optimal Max-3-Cut value.
    num_vertices : int
        Number of vertices.
    num_spins : int
        Total number of spins.
    edges : np.ndarray
        Array of graph edges.

    Returns
    -------
    number_of_successes : float
        Number of successful runs.
    times_success : np.ndarray
        Array of times at which the optimal solution was found for each initialization.
    """
    beta_range = annealing_speed * np.arange(0, max_time+dt, dt)
    number_of_successes = 0.
    times_success = np.zeros(num_inits, dtype=np.float64)-1.
    for idx_init in range(num_inits):
        X = (np.random.rand(num_spins)*2-1)*10**(-10) # random small init
        for idx_beta, beta in enumerate(beta_range):
            X = update_spins_quadratic(X, alpha, beta, J, H, dt)

            # calculate binary energy
            states = amps_to_states_quadratic(X,num_vertices)
            num_wrong_edges = count_wrong_edges(states,edges)
            
            if num_wrong_edges == optimal_solution:
                number_of_successes += 1
                times_success[idx_init] = (idx_beta+1)*dt
                break
            else:
                assert int(num_wrong_edges) > int(optimal_solution)

    return number_of_successes, times_success

def run_quadratic(problem_name, annealing_speed, alpha, B, dir_graphs, dir_solutions, num_inits=100, max_time=10000., dt=0.01, zeta=1.0):
    """
    Run the quadratic Ising formulation simulation for a Max-3-Cut instance.

    Parameters
    ----------
    problem_name : str
        Name of the graph file.
    annealing_speed : float
        Rate of change of beta.
    alpha : float
        Linear gain.
    B : float
        Edge penalty strength.
    dir_graphs : str
        Directory of graph files.
    dir_solutions : str
        Directory of optimal solutions.
    num_inits : int, optional
        Number of runs.
    max_time : float, optional
        Maximum time.
    dt : float, optional
        Time step.
    zeta : float, optional
        Linear term scaling.

    Returns
    -------
    SR : float
        Success rate.
    TTS : float
        Time-to-solution.
    """
    # load problem instance 
    edges, _, num_vertices = read_graph_file(problem_name, dir_graphs)
    optimal_solution = load_optimal_solution(problem_name, dir_solutions)

    # get interaction values for Ising formulation
    num_spins = num_vertices * 3 # each vertex is represented by 3 spins
    J, H = get_interaction_values_quadratic(edges, num_vertices, num_spins, B, zeta=zeta)

    # perform Euler integration
    number_of_successes, times_success = euler_integration_quadratic(num_inits, alpha, annealing_speed, J, H, dt, max_time, optimal_solution, num_vertices, num_spins, edges)

    # calculate SR and TTS
    SR = number_of_successes/num_inits
    TTS = calc_TTS(times_success, dt, max_time, num_inits)
    return SR, TTS
