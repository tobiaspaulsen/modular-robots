import time
from collections.abc import Callable
from deap import base
from deap import tools
from tqdm import tqdm
import numpy as np
import queue
import random
from multiprocessing import (
    connection,
)  # Has to be here to avoid threading import bug...
from threading import Thread

from robot.individual import Individual
from controllers.controller import Controller
from controllers.coupled_oscillator import CoupledOscillator
from evolutionary_algorithms.coevolution import Coevolution
from evaluation.evaluator import Evaluator


def remove_tournament_selection(population: list, population_size: int, tournament_size: int):
    while len(population) > population_size:
        tournament = np.random.choice(population, tournament_size, replace=False)
        tournament = sorted(tournament, key=lambda x: x.fitness, reverse=True)
        for ind in tournament[1:]:
            population.remove(ind)
    return population

def pareto_tournament_selection(population: list, population_size: int, tournament_size: int):
    while len(population) > population_size:
        tournament = np.random.choice(population, tournament_size, replace=False)
        tournament = sorted(tournament, key=lambda x: x.fitness, reverse=True)
        pareto_front = [tournament[0]]
        for ind in tournament[1:]:
            if ind.morph_age < pareto_front[-1].morph_age:
                pareto_front.append(ind)
            else:
                population.remove(ind)
    return population

class TournamentRemove(Coevolution):
    def __init__(self, evaluation_func: Callable[[Individual], float], controller_class: type[Controller], 
                 controller_mutation_sigma: float, create_simple: bool, tournament_size: int, 
                 parallel_processes: int = 1, no_graphics: bool = True, protection: bool = True):
        super().__init__(evaluation_func, controller_class, 0.33, controller_mutation_sigma,  # Only mutate one controller parameter per module on average
                         1, create_simple, tournament_size, parallel_processes, no_graphics)
        self.protection = protection
        if protection:
            self.toolbox.register("select", pareto_tournament_selection, tournament_size=tournament_size)
        else:
            self.toolbox.register("select", remove_tournament_selection, tournament_size=tournament_size)

    def spec_dict(self) -> dict:
        spec_dict = super().spec_dict()
        if self.protection:
            spec_dict["evolution"] = "tournament-remove protection"
        else:
            spec_dict["evolution"] = "tournament-remove no protection"
        return spec_dict

    def evaluate(self, inds):
        if self.parallel_processes == 1:
            try:
                for ind in tqdm(inds, desc="Evaluating Population"):
                    ind.fitness = self.toolbox.evaluate(self.evaluators[0], ind)
            except KeyboardInterrupt:
                print("\nEvaluation interrupted.")
                self.interrupted = True
        else:
            threads = []
            try:
                ind_queue = queue.Queue()
                for g in inds:
                    ind_queue.put(g)

                for i in range(self.parallel_processes):
                    thread = Thread(target=self.evaluate_parallel, args=(ind_queue, self.evaluators[i]))
                    threads.append(thread)
                    thread.start()

                for t in threads:
                    t.join()

            except KeyboardInterrupt:
                print("\nEvaluation interrupted, wait for threads to terminate.")
                self.interrupted = True
                for t in tqdm(threads):
                    t.join()

        self.hall_of_fame.update(inds)

    def step(self, elitism: int = 0):
        timer = time.time()
        self.generation += 1
        parents = list(map(self.toolbox.clone, self.population))
        offspring = list(map(self.toolbox.clone, self.population))

        for ind in offspring:
            if random.random() < 0.5:
                self.toolbox.mutate_controller(ind)
            else:
                self.toolbox.mutate_body(ind)

        for ind in parents:
            ind.morph_age += 1

        self.evaluate(offspring)  # Only offspring has to be evaluated
        self.population = self.toolbox.select(parents + offspring, self.population_size)

        self.diversity_features.append([ind.get_diversity_features() for ind in self.population])
        self.joint_tables.append([ind.build_joint_table() for ind in self.population])
        self.fitnesses_of_each_gen.append([ind.fitness for ind in self.population])
        self.best_of_each_gen.append(self.toolbox.get_best(self.population, k=1)[0])
        record = self.stats.compile(self.population)
        timer = time.time() - timer
        ages = [ind.morph_age for ind in self.population]
        number_of_modules = [len(ind.modules) for ind in self.population]
        average_age = np.mean(ages)
        average_modules = np.mean(number_of_modules)
        std_modules = np.std(number_of_modules)
        self.logbook.record(gen=self.generation, avg_age=average_age, modules=average_modules,
                            std_modules=std_modules, time=timer, **record)
        top20 = self.toolbox.get_best(self.population, k=20)
        fitnesses_ages = [[ind.fitness, ind.morph_age] for ind in top20]
        self.fitness_and_ages_of_top20_per_gen.append(fitnesses_ages)
