import numpy as np
from numba import njit

@njit
def TTS_helper_func(SR, T):
    """
    Time-to-solution (TTS) for a given time window T, as defined in Eq. (10) of https://arxiv.org/pdf/2508.00565v2.
    This function is later optimized over T to find the minimal TTS.
    Parameters
    ----------
    SR : float
        Success rate for the time window T.
    T : float
        Time window (hyperparameter of the TTS definition).
    Returns
    -------
    float
        Time-to-solution for the time window T.
    """
    if SR >= 0.99:
        return T
    elif SR < 1e-10:
        return np.inf
    else:
        return T * np.log10(0.01) / np.log10(1.-SR)

@njit
def SR_helper_func(time_of_GS_list, T, num_inits):
    """
    Success rate (SR) for a given time window T.
    Parameters
    ----------
    time_of_GS_list : np.ndarray
        Array of times at which the optimal solution was found for each initialization.
    T : float
        Time window (hyperparameter of the TTS definition).
    num_inits : int
        Total number of initializations (independent runs).
    Returns
    -------
    float
        Success rate for the time window T.
    """
    return np.float64(np.sum(time_of_GS_list <= T)) / np.float64(num_inits)

@njit
def calc_TTS(times_success, dt, max_time, num_inits):
    """
    Calculate the minimal time-to-solution (TTS) by optimizing over time windows T.
    Parameters
    ----------
    times_success : np.ndarray
        Array of times at which the optimal solution was found for each initialization.
    dt : float
        Time step for Euler integration.
    max_time : float
        Maximum time for Euler integration.
    num_inits : int
        Total number of initializations (independent runs).
    Returns
    -------
    float
        Minimal time-to-solution (TTS).
    """
    times_success = times_success[times_success >= 0.]  # filter out unsuccessful runs
    if len(times_success) == 0:
        return np.inf
    else:
        minimal_TTS = np.inf
        for t in np.arange(0, max_time, dt):
            SR = SR_helper_func(times_success, t, num_inits)
            attempted_TTS = TTS_helper_func(SR, t)
            if attempted_TTS < minimal_TTS:
                minimal_TTS = attempted_TTS
        return minimal_TTS