# -----------------------------------------------------------------------------
# Authors: Maryam Mousavi & Setayesh Asadian Feili
# اعضای گروه: مریم موسوی و ستایش اسدیان فیلی
# Course: Artificial Intelligence – Project 2 (Local Search)
# University of Isfahan
# -----------------------------------------------------------------------------

"""
Genetic Algorithm for sensor placement (bonus task).

Individuals are sensor configurations (a list of (x, y) positions whose
length may vary up to max_sensors).  We use:

    * Tournament selection.
    * Uniform crossover that combines sensors from both parents and
      then trims/dedupes to keep the state valid.
    * Mutation that re-uses the LocalSearchBase neighbor operator
      (move / add / remove), which makes mutation behave consistently
      with the other algorithms.
    * Elitism: the best individual of each generation is copied
      unchanged into the next one.
"""

import random

from search.local_search_base import LocalSearchBase


class GeneticAlgorithm(LocalSearchBase):

    def run(self, initial_state,
            population_size=30,
            generations=100,
            mutation_rate=0.3,
            tournament_k=3,
            elitism=2,
            **kwargs):
        """
        Parameters
        ----------
        initial_state : list[tuple]
            One member of the initial population (the rest are random).
        population_size : int
            Number of individuals per generation.
        generations : int
            Number of generations to evolve.
        mutation_rate : float
            Probability that each child is mutated (a single neighbour
            step) after crossover.
        tournament_k : int
            Tournament size used by selection.
        elitism : int
            Number of best individuals carried over to the next gen.
        """
        # ----- Initial population -----
        population = [list(initial_state)]
        while len(population) < population_size:
            population.append(self.initialize_state())

        def fitness(ind):
            # GA convention: higher is better; we negate the cost.
            return -self.evaluate(ind)

        best_state = min(population, key=self.evaluate)
        best_cost  = self.evaluate(best_state)

        evaluations    = [best_cost]
        states_history = [list(best_state)]

        for _ in range(generations):

            # Sort by cost (ascending => best first).
            population.sort(key=self.evaluate)
            new_population = [list(ind) for ind in population[:elitism]]

            # Fill the rest with offspring.
            while len(new_population) < population_size:
                p1 = self._tournament(population, tournament_k, fitness)
                p2 = self._tournament(population, tournament_k, fitness)
                child = self._crossover(p1, p2)

                if random.random() < mutation_rate:
                    child = self.get_neighbor(child)

                new_population.append(child)

            population = new_population

            # Track best-so-far.
            gen_best = min(population, key=self.evaluate)
            gen_cost = self.evaluate(gen_best)
            if gen_cost < best_cost:
                best_cost  = gen_cost
                best_state = list(gen_best)

            evaluations.append(best_cost)
            states_history.append(list(best_state))

        return best_state, best_cost, evaluations, states_history

    # ------------------------------------------------------------------ #
    #                          Helpers                                   #
    # ------------------------------------------------------------------ #
    def _tournament(self, population, k, fitness):
        """Pick `k` random individuals and return the fittest one."""
        contenders = random.sample(population, min(k, len(population)))
        return max(contenders, key=fitness)

    def _crossover(self, p1, p2):
        """
        Uniform crossover for variable-length sensor sets.

        Steps:
            1. Pool the union of both parents.
            2. Decide the child's size (random value between the two
               parents' sizes).
            3. Sample sensors from the pool without replacement; if the
               pool is too small, fill the rest with random valid cells.
        """
        pool = list({*p1, *p2})
        random.shuffle(pool)

        size = random.randint(min(len(p1), len(p2)),
                              max(len(p1), len(p2)))
        size = max(1, min(size, self.world.max_sensors))

        child = pool[:size]

        # Pad with random positions if we did not get enough.
        used = set(child)
        while len(child) < size:
            from env.utils import random_position
            pos = random_position(self.world)
            if pos not in used:
                child.append(pos)
                used.add(pos)

        return child
