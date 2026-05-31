"""
Hill Climbing search.

This is a stochastic steepest-ascent variant:

    * At each iteration we sample `k_neighbors` neighbours
      (using the shared get_neighbors helper) and pick the one with
      the lowest cost.
    * Strictly better neighbour -> move there.
    * Equal-cost (sideways) neighbour -> move there for a limited number
      of consecutive steps. This helps cross flat plateaus that are very
      common with discrete coverage costs.
    * No improvement and sideways budget exhausted -> RANDOM RESTART
      from a brand new random state, while remembering the best result
      seen so far.

The reason we do random restarts here is that vanilla Hill Climbing is
notoriously bad at escaping local optima; adding restarts inside the
same call lets the algorithm explore several "basins" without changing
the public API of `run`.
"""

from search.local_search_base import LocalSearchBase


class HillClimbing(LocalSearchBase):

    def run(self, initial_state,
            max_iterations=500,
            k_neighbors=25,
            max_sideways=10,
            max_restarts=3,
            **kwargs):
        """
        Parameters
        ----------
        initial_state : list[tuple]
            Starting configuration of sensors.
        max_iterations : int
            Total number of iterations across all restarts.
        k_neighbors : int
            Number of neighbours sampled at every iteration.
        max_sideways : int
            Maximum consecutive equal-cost moves allowed before treating
            the current point as a local optimum.
        max_restarts : int
            How many times we are allowed to restart from a brand new
            random state when we get stuck.
        """
        current_state = list(initial_state)
        current_cost  = self.evaluate(current_state)

        best_state = list(current_state)
        best_cost  = current_cost

        evaluations    = [current_cost]
        states_history = [list(current_state)]

        sideways_used = 0
        restarts_used = 0

        for _ in range(max_iterations):

            # Sample neighbours and pick the best one.
            neighbors  = self.get_neighbors(current_state, k=k_neighbors)
            best_n     = min(neighbors, key=self.evaluate)
            best_n_cst = self.evaluate(best_n)

            if best_n_cst < current_cost:
                # Strictly improving move.
                current_state = best_n
                current_cost  = best_n_cst
                sideways_used = 0

            elif best_n_cst == current_cost and sideways_used < max_sideways:
                # Sideways move: lets us cross a flat plateau.
                current_state = best_n
                sideways_used += 1

            else:
                # We are stuck.  Either restart or stop.
                if restarts_used < max_restarts:
                    restarts_used += 1
                    sideways_used  = 0
                    current_state  = self.initialize_state()
                    current_cost   = self.evaluate(current_state)
                else:
                    # Record final position and exit.
                    evaluations.append(current_cost)
                    states_history.append(list(current_state))
                    break

            # Track best-ever solution (random restarts may move us
            # to a worse basin temporarily).
            if current_cost < best_cost:
                best_cost  = current_cost
                best_state = list(current_state)

            evaluations.append(current_cost)
            states_history.append(list(current_state))

        # Make sure the very last state shown is the BEST one — this
        # is what gets rendered as the "final configuration".
        evaluations.append(best_cost)
        states_history.append(list(best_state))

        return best_state, best_cost, evaluations, states_history
