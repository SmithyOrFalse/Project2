# Project 2 – Local Search (Sensor Placement)

University of Isfahan – AI course – Spring 1404-1405
Dr. Faria Nasiri Mofakham

## Quick start

```bash
pip install -r reqirements.txt
python main.py
```

This runs all five algorithms on `map1` and shows:

* Final cost / state in the terminal
* A matplotlib comparison plot
* A pygame animation of each algorithm's history

To try other maps, edit `main.py` and change `GridWorld("map1")` to `map2 … map7`.

## What was implemented

| File                              | What's inside                                                       |
|-----------------------------------|---------------------------------------------------------------------|
| `search/local_search_base.py`     | Shared cost, neighbor (move/add/remove), and init logic             |
| `search/hill_climbing.py`         | Steepest-ascent + sideways moves + random restarts                  |
| `search/simulated_annealing.py`   | Classic SA with geometric cooling                                   |
| `search/genetic_algorithm.py`     | Bonus – tournament selection, uniform crossover, mutation, elitism  |
| `search/beam_search.py`           | Bonus – local beam search                                           |
| `search/tabu_search.py`           | Bonus – tabu list + aspiration criterion                            |
| `main.py`                         | Entry point – runs all algorithms from a shared initial state       |

The full Persian report (`report.pdf`) explains the cost function, neighborhood
design, parameter choices, and includes experimental results across five maps.

## Project structure

```
project/
├─ main.py                   # entry point
├─ utils.py                  # provided – visualization helpers
├─ reqirements.txt           # provided – pygame, numpy, matplotlib
├─ env/                      # provided – grid world, maps, renderer
├─ search/                   # OUR implementations
└─ report.pdf                # Persian report (deliverable)
```

## Authors

* Maryam Mousavi (مریم موسوی)
* Setayesh Asadian Feili (ستایش اسدیان فیلی)

Repository: https://github.com/SmithyOrFalse/Project2
