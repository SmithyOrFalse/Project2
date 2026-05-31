# -----------------------------------------------------------------------------
# Authors: Maryam Mousavi & Setayesh Asadian Feili
# اعضای گروه: مریم موسوی و ستایش اسدیان فیلی
# Course: Artificial Intelligence – Project 2 (Local Search)
# University of Isfahan
# -----------------------------------------------------------------------------

"""
Local Beam Search (bonus task).

We keep `beam_width` candidate states in parallel; at every iteration we
expand each one into a handful of neighbours, pool everything together,
and keep the `beam_width` lowest-cost states for the next round.

Compared with multiple independent Hill Climbings, beam search lets
"good information found in one beam" propagate to the others, because
all beams compete in the same selection step.
"""

from search.local_search_base import LocalSearchBase


class BeamSearch(LocalSearchBase):

    def run(self, initial_state,
            beam_width=5,
            max_iterations=200,
            neighbors_per_state=10,
            **kwargs):
        """
        Parameters
        ----------
        beam_width : int
            How many states are kept in parallel.
        max_iterations : int
            Number of expansion rounds.
        neighbors_per_state : int
            How many neighbours are generated per kept state at each round.
        """
        # --- Initial beam: the given state + (beam_width-1) random states.
        beam = [list(initial_state)]
        while len(beam) < beam_width:
            beam.append(self.initialize_state())

        best_state = min(beam, key=self.evaluate)
        best_cost  = self.evaluate(best_state)

        evaluations    = [best_cost]
        states_history = [list(best_state)]

        for _ in range(max_iterations):
            # Expand: produce all neighbours from all current beams.
            candidates = []
            for state in beam:
                candidates.extend(
                    self.get_neighbors(state, k=neighbors_per_state)
                )

            # We also keep the previous beam so we never go backwards
            # if all the children happened to be worse (this is a form
            # of elitism).
            candidates.extend(beam)

            # Select the `beam_width` lowest-cost states.
            candidates.sort(key=self.evaluate)
            beam = candidates[:beam_width]

            # Track best-so-far.
            beam_best = beam[0]
            beam_cost = self.evaluate(beam_best)
            if beam_cost < best_cost:
                best_cost  = beam_cost
                best_state = list(beam_best)

            evaluations.append(best_cost)
            states_history.append(list(best_state))

        return best_state, best_cost, evaluations, states_history
