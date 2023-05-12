import time
from collections.abc import Callable
from deap import base
from deap import tools
from tqdm import tqdm
import numpy as np
import queue
from multiprocessing import (
    connection,
)  # Has to be here to avoid threading import bug...
from threading import Thread

import config
from robot.individual import Individual
from controllers.controller import Controller
from evaluation.evaluator import Evaluator


class EA:
    def __init__(self, evaluation_func: Callable[[Individual], float], controller_class: type[Controller],
                 mutation_rate: float, mutation_sigma: float, robot_config_path: str, tournament_size: int = 3,
                 parallel_processes: int = 1, no_graphics: bool = True):
        self.controller_mutation = mutation_rate
        self.controller_sigma = mutation_sigma
        self.body_mutation = 0
        self.tournament_size = tournament_size
        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", Individual, controller_class, json_path=robot_config_path)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", evaluation_func)
        self.toolbox.register("mutate_controller",
                              Individual.mutate_controller,
                              mutation_rate=mutation_rate,
                              mutation_sigma=mutation_sigma)
        self.toolbox.register("select", tools.selTournament, tournsize=tournament_size)
        self.toolbox.register("get_best", tools.selBest, fit_attr="fitness")

        self.stats = tools.Statistics(key=lambda ind: ind.fitness)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("q1", np.quantile, q=0.25)
        self.stats.register("median", np.median)
        self.stats.register("q3", np.quantile, q=0.75)
        self.stats.register("max", np.max)

        self.logbook = tools.Logbook()
        self.logbook.header = "gen", "avg_age", "modules", "min", "median", "max", "time"

        self.hall_of_fame = tools.HallOfFame(1)

        self.population = []
        self.parallel_processes = parallel_processes
        self.generation = 0
        self.evaluators = [Evaluator(no_graphics=no_graphics, editor_mode=False) for _ in range(parallel_processes)]
        self.diversity_features = []
        self.joint_tables = []
        self.fitnesses_of_each_gen = []
        self.fitness_and_ages_of_top20_per_gen = []
        self.best_of_each_gen = []
        self.interrupted = False

    def spec_dict(self) -> dict:
        return {"evolution": "only_controller",
                "controller mutation": self.controller_mutation,
                "controller sigma": self.controller_sigma,
                "body mutation": self.body_mutation,
                "create simple": False,
                "elitism": self.elitism,
                "tournament size": self.tournament_size,
                "population_size": self.population_size,
                "generations": self.generations,
                "evaluation steps": config.EVALUATION_STEPS}


    def evaluate_population(self):
        if self.parallel_processes == 1:
            try:
                for ind in tqdm(self.population, desc="Evaluating Population"):
                    ind.fitness = self.toolbox.evaluate(self.evaluators[0], ind)
            except KeyboardInterrupt:
                print("\nEvaluation interrupted.")
                self.interrupted = True
        else:
            threads = []
            try:
                ind_queue = queue.Queue()
                for g in self.population:
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

        self.hall_of_fame.update(self.population)

    def evaluate_parallel(self, ind_queue: queue.Queue, evaluator: Evaluator):
        while not ind_queue.empty() and not self.interrupted:
            ind = ind_queue.get()
            ind.fitness = self.toolbox.evaluate(evaluator, ind)

    def reset(self, population_size: int):
        timer = time.time()
        self.generation = 0
        self.population_size = population_size
        self.population = self.toolbox.population(n=population_size)

        self.logbook = tools.Logbook()
        self.logbook.header = "gen", "avg_age", "modules", "min", "median", "max", "time"
        self.hall_of_fame = tools.HallOfFame(1)
        self.diversity_features = []
        self.joint_tables = []
        self.fitnesses_of_each_gen = []
        self.best_of_each_gen = []
        self.fitness_and_ages_of_top20_per_gen = []

        self.evaluate_population()
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
        

    def step(self, elitism: int = 0):
        timer = time.time()
        self.generation += 1
        offspring = self.toolbox.select(self.population, self.population_size - elitism)
        offspring = list(map(self.toolbox.clone, offspring))
        elites = self.toolbox.get_best(self.population, k=elitism)

        for ind in offspring:
            self.toolbox.mutate_controller(ind)

        self.population[:] = offspring + elites
        self.evaluate_population()
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

    def run(self, population_size: int, n_generations: int, elitism: int = 0, close_envs: bool = True):
        self.reset(population_size)
        self.elitism = elitism
        self.generations = n_generations
        print(self.logbook.stream)

        for _ in range(1, n_generations):
            if self.interrupted:
                break
            self.step(elitism)
            print(self.logbook.stream)
        if close_envs:
            for evaluator in self.evaluators:
                evaluator.close_env()
