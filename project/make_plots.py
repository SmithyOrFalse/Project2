"""Generate convergence-curve plots used in the report PDF."""
import random
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env.grid_world import GridWorld
from search.hill_climbing       import HillClimbing
from search.simulated_annealing import SimulatedAnnealing
from search.genetic_algorithm   import GeneticAlgorithm
from search.beam_search         import BeamSearch
from search.tabu_search         import TabuSearch

algos = [
    ("Hill Climbing",       HillClimbing),
    ("Simulated Annealing", SimulatedAnnealing),
    ("Genetic Algorithm",   GeneticAlgorithm),
    ("Beam Search",         BeamSearch),
    ("Tabu Search",         TabuSearch),
]


def plot_for_map(map_name, out_path):
    world = GridWorld(map_name)
    random.seed(0)
    init = HillClimbing(world).initialize_state()

    plt.figure(figsize=(9, 5))
    for name, cls in algos:
        random.seed(42)
        algo = cls(world)
        _, _, evals, _ = algo.run(init)
        plt.plot(evals, label=name, alpha=0.85)

    plt.title(f"Convergence of cost - {map_name}")
    plt.xlabel("Iteration")
    plt.ylabel("Cost (lower is better)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved {out_path}")


os.makedirs("report_assets", exist_ok=True)
plot_for_map("map1", "report_assets/convergence_map1.png")
plot_for_map("map6", "report_assets/convergence_map6.png")
plot_for_map("map7", "report_assets/convergence_map7.png")
