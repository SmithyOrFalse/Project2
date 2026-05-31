"""
Base class for all local search algorithms.

Contains the shared logic that every algorithm in this project relies on:
    * evaluate(state)      -> the cost function (lower is better).
    * get_neighbor(state)  -> a single random neighbor produced by move/add/remove.
    * get_neighbors(state) -> a (sampled) list of neighbors, useful for steepest-ascent
                              variants such as Hill Climbing.
    * initialize_state()   -> a valid random starting configuration.

Design notes
------------
1. The cost function balances two objectives:
       (a) maximize the number of targets covered  -> dominant term
       (b) minimize the number of sensors used     -> small penalty per sensor
   In addition, an explicit overlap penalty is included so two sensors covering
   the same target give marginal extra benefit and the search is encouraged
   to spread sensors out across the map.

2. The neighbor function supports the three operations required by the
   project description:
       * MOVE   - pick a sensor and shift it to a neighbouring (or random) cell.
       * ADD    - add a new sensor at a free, valid cell  (if the cap allows).
       * REMOVE - remove an existing sensor (if at least one remains).
   This keeps the size of the state dynamic, which is essential for the
   trade-off between coverage quality and number of sensors used.
"""

import random
from env.utils import random_position


# --------------------------------------------------------------------------- #
#                       Tunable weights for the cost                          #
# --------------------------------------------------------------------------- #
# Each uncovered target is much more expensive than an extra sensor,
# so the search will first try to cover every target and then try to
# shrink the sensor set. Numbers were chosen so that uncovered targets
# always dominate the other terms.
W_UNCOVERED = 100   # penalty per uncovered target           (primary objective)
W_SENSOR    = 5     # penalty per sensor placed              (trade-off term)
W_OVERLAP   = 1     # penalty per extra sensor covering a target already covered


class LocalSearchBase:

    def __init__(self, world):
        self.world = world

    # --------------------------------------------------------------------- #
    #                       Coverage / cost                                 #
    # --------------------------------------------------------------------- #
    def _covered_targets(self, state):
        """
        Return a dict { target_position : number_of_sensors_covering_it }.

        A target at (tx, ty) is considered covered by a sensor at (sx, sy)
        when the Manhattan distance between them is <= sensor_range
        (this is the same metric the renderer uses).
        """
        sensor_range = self.world.sensor_range
        coverage = {}

        for tx, ty in self.world.get_targets():
            count = 0
            for sx, sy in state:
                if abs(sx - tx) + abs(sy - ty) <= sensor_range:
                    count += 1
            coverage[(tx, ty)] = count

        return coverage

    def evaluate(self, state):
        """
        Cost function (lower is better).

            cost = W_UNCOVERED * (uncovered targets)
                 + W_SENSOR    * (number of sensors)
                 + W_OVERLAP   * (extra coverers beyond the first per target)

        The dominant term is the uncovered-target penalty so the optimizer
        is pushed to cover every target first.  The sensor penalty creates
        the trade-off requested in the project, and the overlap term
        discourages clustering sensors on the same target when the search
        could instead spread them out.
        """
        coverage = self._covered_targets(state)

        uncovered = sum(1 for c in coverage.values() if c == 0)
        overlap   = sum(max(c - 1, 0) for c in coverage.values())
        n_sensors = len(state)

        return (W_UNCOVERED * uncovered
                + W_SENSOR  * n_sensors
                + W_OVERLAP * overlap)

    # --------------------------------------------------------------------- #
    #                       Initialization                                  #
    # --------------------------------------------------------------------- #
    def initialize_state(self, n_sensors=None):
        """
        Generate a random, valid starting configuration.

        By default we start from roughly half of the available sensor budget
        so the search has room to both add and remove sensors during
        exploration.  All sensors are placed on distinct valid cells.
        """
        if n_sensors is None:
            # Start from a moderate size so both grow and shrink moves are useful.
            n_sensors = max(1, self.world.max_sensors // 2)

        n_sensors = min(n_sensors, self.world.max_sensors)

        state = []
        used = set()
        attempts = 0
        # Cap attempts so we never loop forever on very dense maps.
        while len(state) < n_sensors and attempts < 1000:
            pos = random_position(self.world)
            if pos not in used:
                state.append(pos)
                used.add(pos)
            attempts += 1

        return state

    # --------------------------------------------------------------------- #
    #                       Neighbor generation                             #
    # --------------------------------------------------------------------- #
    def _move_sensor(self, state):
        """Return a new state where one sensor is moved to a different cell."""
        if not state:
            return state

        new_state = list(state)
        idx = random.randrange(len(new_state))
        used = set(new_state)

        # Try a small local shift first (4-neighbour), fall back to a
        # uniformly random valid cell.  The local shift helps fine-tuning
        # while the random fallback helps escape local optima.
        sx, sy = new_state[idx]
        local_candidates = [
            (sx + dx, sy + dy)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        ]
        random.shuffle(local_candidates)

        for nx, ny in local_candidates:
            if (self.world.is_valid_position(nx, ny)
                    and (nx, ny) not in used):
                new_state[idx] = (nx, ny)
                return new_state

        # Fallback: jump to a random valid free cell.
        for _ in range(50):
            pos = random_position(self.world)
            if pos not in used:
                new_state[idx] = pos
                return new_state

        return new_state  # nothing better found; return unchanged copy

    def _add_sensor(self, state):
        """Return a new state with one extra sensor (if the cap allows)."""
        if len(state) >= self.world.max_sensors:
            return state

        new_state = list(state)
        used = set(new_state)
        for _ in range(50):
            pos = random_position(self.world)
            if pos not in used:
                new_state.append(pos)
                return new_state
        return new_state

    def _remove_sensor(self, state):
        """Return a new state with one sensor removed (if at least one remains)."""
        if len(state) <= 1:
            return state

        new_state = list(state)
        idx = random.randrange(len(new_state))
        new_state.pop(idx)
        return new_state

    def get_neighbor(self, state):
        """
        Return ONE random neighbour by randomly choosing among the three
        operations.  Move is given a higher weight because it is by far the
        most useful operation once the rough sensor count is right.
        """
        # Build the list of admissible operations for this state.
        ops = ['move'] * 3 + ['add'] + ['remove']

        # Guard against trying to add when we are at the cap, or trying
        # to remove when we only have one sensor.
        if len(state) >= self.world.max_sensors:
            ops = [o for o in ops if o != 'add']
        if len(state) <= 1:
            ops = [o for o in ops if o != 'remove']

        op = random.choice(ops)
        if op == 'move':
            return self._move_sensor(state)
        if op == 'add':
            return self._add_sensor(state)
        return self._remove_sensor(state)

    def get_neighbors(self, state, k=20):
        """
        Sample a list of `k` neighbours.  Used by steepest-ascent Hill
        Climbing as an approximation of the full neighbourhood (the true
        neighbourhood is huge because of the move-anywhere operator).
        """
        return [self.get_neighbor(state) for _ in range(k)]
