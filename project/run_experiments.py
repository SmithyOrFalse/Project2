# -----------------------------------------------------------------------------
# Authors: Maryam Mousavi & Setayesh Asadian Feili
# اعضای گروه: مریم موسوی و ستایش اسدیان فیلی
# Course: Artificial Intelligence – Project 2 (Local Search)
# University of Isfahan
# -----------------------------------------------------------------------------

"""Run all algorithms on all maps, multiple seeds, collect statistics."""
import json
import random
import statistics
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")

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

maps_to_test = ["map1", "map3", "map4", "map6", "map7"]
N_SEEDS = 3

results = {}

for m in maps_to_test:
    world = GridWorld(m)
    n_targets = len(world.get_targets())
    results[m] = {
        "rows": world.rows, "cols": world.cols,
        "range": world.sensor_range,
        "max_sensors": world.max_sensors,
        "n_targets": n_targets,
        "runs": {},
    }
    print(f"\n=== {m}  targets={n_targets}  range={world.sensor_range}  "
          f"max_sensors={world.max_sensors} ===")
    for name, cls in algos:
        costs, sensors, coverages = [], [], []
        for seed in range(N_SEEDS):
            random.seed(seed)
            init = HillClimbing(world).initialize_state()
            random.seed(seed * 100 + 1)
            algo = cls(world)
            state, cost, evals, hist = algo.run(init)

            # Compute coverage percentage explicitly.
            sr = world.sensor_range
            covered = 0
            for tx, ty in world.get_targets():
                for sx, sy in state:
                    if abs(sx - tx) + abs(sy - ty) <= sr:
                        covered += 1
                        break
            coverages.append(covered / max(n_targets, 1))
            costs.append(cost)
            sensors.append(len(state))

        results[m]["runs"][name] = {
            "cost_mean":     statistics.mean(costs),
            "cost_min":      min(costs),
            "cost_stdev":    statistics.stdev(costs) if len(costs) > 1 else 0,
            "sensors_mean":  statistics.mean(sensors),
            "coverage_mean": statistics.mean(coverages),
        }
        print(f"  {name:22s} cost(mean)={statistics.mean(costs):7.1f}  "
              f"best={min(costs):6.1f}  "
              f"cov={statistics.mean(coverages)*100:5.1f}%  "
              f"sensors={statistics.mean(sensors):4.1f}")

with open("experiments.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved -> experiments.json")
