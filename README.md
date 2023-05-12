# Modular-Robots
Repository for master's thesis by Tobias Paulsen (tobiasrp@uio.no)

## Evolutionary Algorithms:
- ``age_fitness_pateto.py``: Tournament-add based on both age and fitness (morphological protection)
- ``bins_afp.py``: Same as above just with bins of ages
- ``cheney.py``: Very elitist approach used by Cheney et al. (possible with and without protection)
- ``coevolution.py``: Standard co-evolution of control and morphology with tournament-add
- ``increasing_tournament.py``: Same as age_fitness_pareto with a gradually increasing tournament size
- ``only_controller.py``: Base EA, only evolves the control system of a modular robot
- ``tournament_remove.py``: Algorithm used in the thesis. Tournament-remove based on either age and fitness or just fitness.
