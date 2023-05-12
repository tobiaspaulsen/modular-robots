import os
import pickle
from copy import deepcopy

from evaluation.evaluator import Evaluator
from robot.individual import Individual
import config


def load_and_evaluate_best(experiment_folder: str, eval_steps: int, editor_mode: bool = False):
    with open(f"{experiment_folder}/hall_of_fame.pickle", "rb") as f:
        individuals = pickle.load(f)
        evaluator = Evaluator(no_graphics=False, editor_mode=editor_mode)
        best_ind = individuals[0]
        print(f"Old fitness: {best_ind.fitness}")
        fitness = evaluator.evaluate(best_ind, eval_steps=eval_steps)
        print(f"Fitness: {fitness}")
        evaluator.close_env()


def load_and_evaluate_several(folders: list[str], eval_steps: int, editor_mode: bool = False, env: str = None):
    folders.sort()
    if env is not None:
        if os.path.exists(f"{config.UNITY_BUILD_BASE_PATH}/{env}.app"):
            config.UNITY_BUILD_PATH = f"{config.UNITY_BUILD_BASE_PATH}/{env}"
        elif os.path.exists(f"{config.UNITY_BUILD_BASE_PATH}/{env}/{env}.x86_64"):
            config.UNITY_BUILD_PATH = f"{config.UNITY_BUILD_BASE_PATH}/{env}/{env}"
    individuals = []
    for folder in folders:
        with open(f"{folder}/hall_of_fame.pickle", "rb") as f:
            hall_of_fame = pickle.load(f)
            individuals.append((folder, hall_of_fame[0]))
    
    individuals.sort(key=lambda x: x[1].fitness, reverse=True)

    evaluator = Evaluator(no_graphics=False, editor_mode=editor_mode)
    for folder, ind in individuals:
        print(f"Old fitness: {ind.fitness}")
        fitness = evaluator.evaluate(ind, eval_steps=eval_steps)
        print(f"Fitness: {fitness}")
    evaluator.close_env()


def load_and_evaluate_record(experiment_folder: str, eval_steps: int, editor_mode: bool = False):
    individuals = []
    with open(f"{experiment_folder}/best_of_each_gen.pickle", "rb") as f:
        hall_of_fame = pickle.load(f)
        ind = hall_of_fame[-1]
        for i in ind.record:
            individuals.append(i)
        individuals.append(ind)
    evaluator = Evaluator(no_graphics=False, editor_mode=editor_mode)
    for i in range(0, len(individuals), 5):
        fitness = evaluator.evaluate(individuals[i][1], eval_steps=eval_steps)
        print(f"Fitness:", fitness)
    
    evaluator.close_env()
