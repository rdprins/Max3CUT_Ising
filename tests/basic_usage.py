import os
import max3cut_ising

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DIR_GRAPHS = os.path.join(BASE_DIR, "graphs")
DIR_SOLUTIONS = os.path.join(BASE_DIR, "solutions")

# example run Ising formulation
SR, TTS = max3cut_ising.run_quadratic("g05_30.0", 0.001, -5., 0.3,
                    DIR_GRAPHS, DIR_SOLUTIONS,
                    max_time=100.) # for quick testing
print(f"Ising formulation: Success rate = {SR:.3f}, Time-to-solution = {TTS:.3f} s")

# example run rescaled Ising formulation
SR, TTS = max3cut_ising.run_quadratic("g05_30.0", 0.001, -5., 1.0,
                    DIR_GRAPHS, DIR_SOLUTIONS,
                    zeta=0.6,
                    max_time=100.) # for quick testing
print(f"Rescaled Ising formulation: Success rate = {SR:.3f}, Time-to-solution = {TTS:.3f} s")

# example run higher-order Ising formulation
SR, TTS = max3cut_ising.run_HO("g05_30.0", 0.001, -5., 0.1,
                    DIR_GRAPHS, DIR_SOLUTIONS,
                    max_time=100.) # for quick testing
print(f"Higher-order formulation: Success rate = {SR:.3f}, Time-to-solution = {TTS:.3f} s")