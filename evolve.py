import pickle
import os
import json
from datetime import date
import numpy as np

from evolutionary_algorithms.only_controller import EA
from evolutionary_algorithms.coevolution import Coevolution
from evolutionary_algorithms.age_fitness_pareto import AgeFitnessPareto
from evolutionary_algorithms.tournament_remove import TournamentRemove
from evolutionary_algorithms.cheney import Cheney
from evolutionary_algorithms.bins_afp import BinsAgeFitnessPareto
from evolutionary_algorithms.increasing_tournament import IncreasingTournament
from evaluation.evaluator import Evaluator
from controllers.coupled_oscillator import CoupledOscillator
import config

def get_run_nr():
    with open(f"{config.RESULTS_PATH}/runs.csv", "r") as file:
        lines = file.readlines()
        if len(lines) == 0:
            last_run_nr = 0
        else:
            last_run_nr = int(lines[-1].split(",")[0])
    run_nr = last_run_nr + 1
    return run_nr


def get_run_folder(run_nr: int = None) -> str:
    if run_nr is None:
        run_nr = get_run_nr()
    # Add current run to runs.csv
    with open(f"{config.RESULTS_PATH}/runs.csv", "a") as file:
        file.write(f"{run_nr}, {date.today().strftime('%d/%m/%y')}\n")
    # Create run folder if it doesn't exist
    folder_name = f"{config.RESULTS_PATH}/run{run_nr}"
    os.makedirs(os.path.dirname(f"{folder_name}/"), exist_ok=True)
    return folder_name


def save_results(ea, folder: str):
    os.makedirs(os.path.dirname(f"{folder}/"), exist_ok=True)
    with open(f"{folder}/logbook.pickle", "wb") as file:
        pickle.dump(ea.logbook, file)
    with open(f"{folder}/hall_of_fame.pickle", "wb") as file:
        pickle.dump(ea.hall_of_fame, file)
    with open(f"{folder}/fitnesses_of_each_gen.pickle", "wb") as file:
        pickle.dump(ea.fitnesses_of_each_gen, file)
    with open(f"{folder}/best_of_each_gen.pickle", "wb") as file:
        pickle.dump(ea.best_of_each_gen, file)
    with open(f"{folder}/last_generation.pickle", "wb") as file:
        pickle.dump(ea.population, file)
    with open(f"{folder}/top20_fitness_age.pickle", "wb") as file:
        pickle.dump(ea.fitness_and_ages_of_top20_per_gen, file)

    # NB: When using pareto-add the population size can vary
    diversity_features = np.asarray(ea.diversity_features, dtype=object)
    np.save(f"{folder}/diversity_features.npy", diversity_features)
    joint_tables = np.asarray(ea.joint_tables, dtype=object)
    np.save(f"{folder}/joint_tables.npy", joint_tables)
   

def evolve(ea: EA, pop_size: int, generations: int, elitism: int, save: bool = True, env: str = None):
    if env is not None:
        if os.path.exists(f"{config.UNITY_BUILD_BASE_PATH}/{env}.app"):
            config.UNITY_BUILD_PATH = f"{config.UNITY_BUILD_BASE_PATH}/{env}"
        elif os.path.exists(f"{config.UNITY_BUILD_BASE_PATH}/{env}/{env}.x86_64"):
            config.UNITY_BUILD_PATH = f"{config.UNITY_BUILD_BASE_PATH}/{env}/{env}"

    ea.run(pop_size, generations, elitism)

    if save:
        folder = get_run_folder()
        with open(f"{folder}/specs.json", "w") as file:
            json.dump(ea.spec_dict(), file)
        save_results(ea, folder)
        

def evolve_n_times(ea: EA, pop_size: int, generations: int, n: int, elitism: int = 0, env: str = None):
    if env is not None:
        if os.path.exists(f"{config.UNITY_BUILD_BASE_PATH}/{env}.app"):
            config.UNITY_BUILD_PATH = f"{config.UNITY_BUILD_BASE_PATH}/{env}"
        elif os.path.exists(f"{config.UNITY_BUILD_BASE_PATH}/{env}/{env}.x86_64"):
            config.UNITY_BUILD_PATH = f"{config.UNITY_BUILD_BASE_PATH}/{env}/{env}"

    folder = ""
    for i in range(n):
        if i == n-1:
            ea.run(pop_size, generations, elitism, close_envs=True)
        else:
            ea.run(pop_size, generations, elitism, close_envs=False)
        if i == 0: # Only want to save stats if at least one run is successfull
            folder = get_run_folder()
            with open(f"{folder}/specs.json", "w") as file:
                json.dump(ea.spec_dict(), file)
        save_results(ea, f"{folder}/{i}")


if __name__ == "__main__":
    ea = TournamentRemove(
        Evaluator.evaluate,
        CoupledOscillator,
        controller_mutation_sigma=0.2,
        create_simple=True,
        tournament_size=2,
        parallel_processes=64,
        no_graphics=True,
        protection=True
    )
    evolve_n_times(ea, pop_size=100, generations=500, n=10, elitism=1, env="flat")
