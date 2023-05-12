from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import (
    ActionTuple
)
import socket
from evaluation.unity_side_channel import CustomSideChannel
import numpy as np
import random

import config
from robot.individual import Individual


HIGHEST_WORKER_ID = 65535 - UnityEnvironment.BASE_ENVIRONMENT_PORT


class Evaluator:
    def __init__(self, no_graphics: bool = True, editor_mode: bool = False):
        self.no_graphics = no_graphics
        self.editor_mode = editor_mode
        self.env = None
        self.channel = CustomSideChannel()

    @staticmethod
    def is_port_in_use(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0

    @staticmethod
    def is_worker_id_open(worker_id: int) -> bool:
        return not Evaluator.is_port_in_use(UnityEnvironment.BASE_ENVIRONMENT_PORT + worker_id)

    @staticmethod
    def get_worker_id() -> int:
        pid = random.randrange(HIGHEST_WORKER_ID)
        while not Evaluator.is_worker_id_open(pid):
            print("Socket is occupied, trying a new worker_id")
            pid = random.randrange(HIGHEST_WORKER_ID)
        return pid

    def get_env(self):
        if self.env is None:
            if self.editor_mode:
                self.env = UnityEnvironment(seed=config.SEED, side_channels=[self.channel],
                                            no_graphics=self.no_graphics) 
            else:
                self.env = UnityEnvironment(file_name=config.UNITY_BUILD_PATH, seed=config.SEED,
                                            side_channels=[self.channel], no_graphics=self.no_graphics,
                                            worker_id=Evaluator.get_worker_id(), log_folder=config.LOG_PATH)
            for _ in range(10):  # Fixes determinism
                self.env.step()
        
        self.env.reset()
        return self.env

    def close_env(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    def evaluate(self, ind: Individual, debug: bool = False, eval_steps: int = config.EVALUATION_STEPS) -> np.float32:
        env = self.get_env()

        ind.reset_controllers()

        json_string = ind.get_json_string()
        self.channel.wait_for_robot_string = True
        self.channel.send_string(json_string)
        # unity waits a few frames before creating the robot (for determinism)
        while self.channel.wait_for_robot_string:
            env.step()
        
        for _ in range(config.WAIT_WHILE_FALLING_STEPS):
            env.step()

        fitness = -1.0
        max_fitness = -1.0
        total_movement = 0.0
        behavior_name = list(env.behavior_specs)[0]
        
        for s in range(eval_steps):
            obs, _ = env.get_steps(behavior_name)
            actions = np.ndarray(shape=(1, config.MAX_MODULES_UNITY), dtype=np.float32)
            actions = ind.get_next_action(actions, config.PYTHON_DELTA_TIME)
            env.set_action_for_agent(behavior_name, obs.agent_id, ActionTuple(actions))
            
            try:
                fitness = obs.reward[0]
            except:
                print("Cannot get fitness")

            total_movement += np.abs(fitness)
            if fitness > max_fitness:
                max_fitness = fitness
            if fitness < -2 or max_fitness - fitness > 1:
                break
            if s > 30 and total_movement < 0.2:
                break
            if fitness > config.MAX_FITNESS:  # If a physics bug occurs to get an impossibly high fitness value
                fitness = max_fitness = 0
                break

            env.step()

        if debug:
            print(f"[Python]: fitness = {fitness}")
        
        if config.CLEAN_UP_GENOMES:
            module_keys = self.channel.created_robot_module_keys
            if len(ind.modules) != len(module_keys):
                ind.clean_up_genome(module_keys)

        return np.round(max_fitness, 3)

