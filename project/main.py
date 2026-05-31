# -----------------------------------------------------------------------------
# Authors: Maryam Mousavi & Setayesh Asadian Feili
# اعضای گروه: مریم موسوی و ستایش اسدیان فیلی
# Course: Artificial Intelligence – Project 2 (Local Search)
# University of Isfahan
# -----------------------------------------------------------------------------

"""
University: University of Isfahan
Faculty: Mathematics and Statistics
Branch: Computer Science
Course: Artificial Intelligence
Professor: Dr. Faria Nasiri Mofakham
TAs: MehrAzin Marzough, Mohammad Karimi, Anahita Honarmandian
Project: Implementing Local Search Algorithms for a Sensor Placement Optimization Problem
"""
import random
import re

import matplotlib
matplotlib.use("TkAgg")

from env.grid_world import GridWorld
from search.hill_climbing import HillClimbing
from search.simulated_annealing import SimulatedAnnealing
from search.genetic_algorithm import GeneticAlgorithm
from search.beam_search import BeamSearch
from search.tabu_search import TabuSearch
from utils import represent


def run_algorithms(world, initial_state, algorithm_classes):
 best_states = []
 best_costs = []
 evaluations = []
 histories = []
 names = []

 for algorithm_class in algorithm_classes:
 # HillClimbing -> "Hill Climbing"
 name = re.sub(r'(?<!^)([A-Z])', r' \1', algorithm_class.__name__)
 names.append(name)

 algorithm_instance = algorithm_class(world)

 print(f"\nRunning {name}...")

 state, cost, evals, hist = algorithm_instance.run(initial_state)

 best_states.append(state)
 best_costs.append(cost)
 evaluations.append(evals)
 histories.append(hist)

 represent(
 best_states=best_states,
 best_costs=best_costs,
 evaluations=evaluations,
 histories=histories,
 names=names,
 world=world,
 )


if __name__ == "__main__":
 # Fix the seed so all algorithms see exactly the same random draws
 # for their starting state - this makes the comparison fair.
 random.seed(42)

 # Load the map. Try map1..map7 to compare behaviour.
 world = GridWorld("map1")

 algorithm_classes = [
 HillClimbing,
 SimulatedAnnealing,
 # ----- Bonus algorithms (comment out to skip) -----
 GeneticAlgorithm,
 BeamSearch,
 TabuSearch,
 ]

 # A single shared starting state for all algorithms (fair comparison).
 # We instantiate any of the search classes just to access initialize_state().
 initial_state = HillClimbing(world).initialize_state()

 run_algorithms(world, initial_state, algorithm_classes)
