# -----------------------------------------------------------------------------
# Authors: Maryam Mousavi & Setayesh Asadian Feili
# اعضای گروه: مریم موسوی و ستایش اسدیان فیلی
# Course: Artificial Intelligence – Project 2 (Local Search)
# University of Isfahan
# -----------------------------------------------------------------------------

"""
Tabu Search (bonus task).

The key idea: at every step we move to the BEST neighbour we have not
visited recently, even if it makes the cost worse.  A short-term memory
(the "tabu list") prevents the search from cycling back into states it
has just left, which is the classical fix for the local-optimum problem
that plain Hill Climbing suffers from.

Implementation details:

    * The tabu list stores a canonical fingerprint of recent states
      (a frozenset of sensor positions).  Frozensets are used so the
      order of the sensors does not matter when comparing two states.
    * `aspiration`: a tabu move IS allowed if it would beat the
      best-known cost (a tabu move that improves the global best is
      considered worth taking anyway).
"""

from collections import deque

from search.local_search_base import LocalSearchBase


class TabuSearch(LocalSearchBase):

    def run(self, initial_state,
            max_iterations=400,
            k_neighbors=20,
            tabu_tenure=20,
            **kwargs):
        """
        Parameters
        ----------
        max_iterations : int
            Total number of iterations.
        k_neighbors : int
            Neighbours sampled per iteration.
        tabu_tenure : int
            Max length of the tabu list (how long a state stays "forbidden").
        """
        current_state = list(initial_state)
        current_cost  = self.evaluate(current_state)

        best_state = list(current_state)
        best_cost  = current_cost

        evaluations    = [current_cost]
        states_history = [list(current_state)]

        tabu = deque(maxlen=tabu_tenure)
        tabu.append(frozenset(current_state))

        for _ in range(max_iterations):
            neighbors = self.get_neighbors(current_state, k=k_neighbors)

            # Pick the best neighbour that is either non-tabu or
            # passes the aspiration criterion.
            chosen      = None
            chosen_cost = None
            for n in neighbors:
                key = frozenset(n)
                n_cost = self.evaluate(n)

                is_tabu        = key in tabu
                pass_aspiration = n_cost < best_cost

                if is_tabu and not pass_aspiration:
                    continue

                if chosen is None or n_cost < chosen_cost:
                    chosen      = n
                    chosen_cost = n_cost

            if chosen is None:
                # Every neighbour is tabu and none passes aspiration.
                # Just take the globally best neighbour and continue;
                # this is rare but keeps the search progressing.
                chosen      = min(neighbors, key=self.evaluate)
                chosen_cost = self.evaluate(chosen)

            current_state = chosen
            current_cost  = chosen_cost
            tabu.append(frozenset(current_state))

            if current_cost < best_cost:
                best_cost  = current_cost
                best_state = list(current_state)

            evaluations.append(current_cost)
            states_history.append(list(current_state))

        # End on the best-ever state.
        evaluations.append(best_cost)
        states_history.append(list(best_state))

        return best_state, best_cost, evaluations, states_history
