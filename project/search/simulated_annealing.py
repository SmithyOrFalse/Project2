"""
Simulated Annealing.

The Metropolis criterion is applied with an exponential cooling
schedule:

        T_{t+1} = T_t * alpha

At each step we generate ONE random neighbour and:

    * If it improves cost                -> always accept.
    * If it is worse by `delta`          -> accept with probability
                                            exp(-delta / T).

Parameters were chosen so that early in the run almost any worsening
move can be accepted (encouraging exploration) and by the end essentially
only improvements are accepted (exploitation), which is the canonical
behaviour of SA and the reason it can usually beat plain Hill Climbing
on landscapes with many local optima.
"""

import math
import random

from search.local_search_base import LocalSearchBase


class SimulatedAnnealing(LocalSearchBase):

    def run(self, initial_state,
            max_iterations=2000,
            initial_temperature=10.0,
            min_temperature=1e-3,
            alpha=0.995,
            **kwargs):
        """
        Parameters
        ----------
        initial_state : list[tuple]
            Starting configuration of sensors.
        max_iterations : int
            Hard cap on the number of iterations.
        initial_temperature : float
            Starting temperature.  Higher -> more exploration.
        min_temperature : float
            Stop early once we have cooled below this.
        alpha : float
            Geometric cooling factor (0 < alpha < 1).  Closer to 1
            means slower cooling and usually better results, at the
            cost of more iterations.
        """
        current_state = list(initial_state)
        current_cost  = self.evaluate(current_state)

        best_state = list(current_state)
        best_cost  = current_cost

        evaluations    = [current_cost]
        states_history = [list(current_state)]

        temperature = initial_temperature

        for _ in range(max_iterations):

            if temperature < min_temperature:
                break

            neighbor      = self.get_neighbor(current_state)
            neighbor_cost = self.evaluate(neighbor)

            delta = neighbor_cost - current_cost

            if delta < 0:
                # Improvement: always accept.
                accept = True
            else:
                # Worsening: accept with Boltzmann probability.  As T
                # shrinks this probability collapses towards 0.
                accept = random.random() < math.exp(-delta / temperature)

            if accept:
                current_state = neighbor
                current_cost  = neighbor_cost

                if current_cost < best_cost:
                    best_cost  = current_cost
                    best_state = list(current_state)

            evaluations.append(current_cost)
            states_history.append(list(current_state))

            # Geometric cooling.
            temperature *= alpha

        # End on the best-ever state so the renderer shows the optimum.
        evaluations.append(best_cost)
        states_history.append(list(best_state))

        return best_state, best_cost, evaluations, states_history
