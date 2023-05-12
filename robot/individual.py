import json
import random
import uuid
import numpy as np
from copy import deepcopy

import config
from controllers.controller import Controller
from controllers.coupled_oscillator import CoupledOscillator
from robot.module import Module, Root, BodyJoint, LimbJoint


class Individual:
    def __init__(self, controller_class: type[Controller], json_path: str = None,
                 fitness=-1.0, create_simple: bool = True):
        self.fitness = fitness
        self.record = []
        self.controller_class = controller_class
        self.prev_age = 1   # Used if the age is reset because of a mutation that doesn't render
        self.added = 0  # Added since last evaluation
        self.morph_age = 0
        self.mutations = []
        # Diversity features:
        self.body_joints = 0
        self.limb_joints = 0
        self.limbs = 0

        if json_path is not None:
            self.load_from_json(json_path)  # Handle without complementary here as well
        else:  # Temporary
            self.root = Root(self.controller_class)
            self.modules = [self.root]
            self.modules_without_complementaries = [self.root]
            self.create(create_simple)

    def create(self, simple: bool):
        joint = BodyJoint(str(uuid.uuid4()), self.root, 2, 0, self.controller_class, random.choice(config.BODY_JOINTS))
        self.root.children.append(joint)
        self.root.number_of_body_children += 1
        self.modules.append(joint)
        self.modules_without_complementaries.append(joint)

        if not simple:
            number_of_modules = random.randint(4, config.MAX_MODULES_PYTHON)
            while len(self.modules) < number_of_modules:
                self.add_module(init=True)

        self.generate_module_lists()

    def load_from_json(self, file_name: str) -> dict:  
        # TODO: Change representation
        # TODO: Handle complementary limbs
        with open(file_name) as f:
            json_dict = json.load(f)

        modules = {}
        nodes = json_dict["nodes"]
        for node in nodes:
            module_type = node["type"]
            if module_type == "Root":
                self.root = Root(self.controller_class)
                modules["root"] = self.root

            elif module_type in config.BODY_JOINTS:
                parent = modules[node["parent"]]
                body_joint = BodyJoint(node["name"], parent, int(node["connection_site"]),
                                       node["angle"], self.controller_class, module_type)
                modules[node["name"]] = body_joint
                parent.children.append(body_joint)
                parent.number_of_body_children += 1

            elif module_type in config.LIMB_JOINTS:
                parent = modules[node["parent"]]
                limb_joint = LimbJoint(node["name"], parent, int(node["connection_site"]),
                                       node["angle"], self.controller_class, module_type)
                modules[node["name"]] = limb_joint
                parent.children.append(limb_joint)
                parent.number_of_limb_children += 1

        self.modules = []
        self.modules_without_complementaries = []
        self.generate_module_lists()

    def get_json_string(self) -> str:
        nodes = [module.get_dict_for_json() for module in self.modules]
        return json.dumps({"nodes": nodes})

    def mutate_controller(self, mutation_rate: float, mutation_sigma: float):
        for module in self.modules:
            module.controller.mutate(mutation_rate, mutation_sigma)
        self.morph_age += 1

    def mutate_body(self, mutation_rate: float, mutations: list = None):
        if mutations == None:
            mutations = ["remove", "add", "swap"]
        elif len(mutations) == 0:
            return

        if random.uniform(0, 1) < mutation_rate:
            clone = deepcopy(self)
            mutation = random.choice(mutations)
            success = eval(f"self.{mutation}_module()")

            if success:
                self.prev_age = self.morph_age
                self.morph_age = 0
                clone.record = []
                self.record.append((self.prev_age, clone))
            else:
                filtered_mutations = list(filter(lambda mut: mut != mutation, mutations))
                self.mutate_body(mutation_rate=1, mutations=filtered_mutations)

    def add_module(self, depth: int = 1, init: bool = False) -> bool:
        if depth > config.MAX_ADD_DEPTH:
            return True
        
        if len(self.modules) >= config.MAX_MODULES_PYTHON:
            return False

        modules_can_add_body = list(filter(Module.can_add_body, self.modules))
        modules_can_add_limb = list(filter(Module.can_add_limb, self.modules_without_complementaries))

        if len(modules_can_add_body) == 0 and len(modules_can_add_limb) == 0:
            return False

        number_of_limb_connectors = 0
        number_of_body_connectors = 0
        for m in self.modules:
            if m.can_add_limb():
                if isinstance(m, LimbJoint):
                    number_of_limb_connectors += 3
                else:
                    number_of_limb_connectors += 2
            if m.can_add_body():
                number_of_body_connectors += 1

        chance_of_body = number_of_body_connectors / (number_of_body_connectors + number_of_limb_connectors)
        if random.uniform(0, 1) < chance_of_body:
            module = random.choice(modules_can_add_body)
            module.add_body(init)
        else:
            module = random.choice(modules_can_add_limb)
            module.add_limb(init)
        
        self.added += 1
        self.generate_module_lists()
        if random.uniform(0, 1) < config.REPEAT_ADD_PROB:
            self.add_module(depth + 1, init)
        return True

    def _remove_module(self, module: Module):
        module.parent.children.remove(module)

        if isinstance(module, LimbJoint):
            module.parent.number_of_limb_children -= 1
            complementary = module.complementary_limb
            complementary.parent.children.remove(complementary)
            complementary.parent.number_of_limb_children -= 1
        else:
            module.parent.number_of_body_children -= 1

    def remove_module(self) -> bool:
        modules = self.modules_without_complementaries[1:]  # Every module except for root and complementary limbs

        if len(modules) == 0:
            return False

        to_remove = random.choice(modules)
        self._remove_module(to_remove)

        self.generate_module_lists()
        return True

    def swap_module(self):
        modules = self.modules_without_complementaries[1:]

        if len(modules) == 0:  # Should not be possible
            return False

        random.choice(modules).swap()
        return True

    def generate_module_lists(self):  # Generates module list based on BFS
        self.modules[:] = [self.root]
        self.modules_without_complementaries[:] = [self.root]
        queue = [self.root]
        self.body_joints = 0
        self.limb_joints = 0
        self.limbs = 0

        while queue:
            m = queue.pop(0)
            for child in m.children:
                if child not in self.modules:
                    self.modules.append(child)
                    queue.append(child)
                    if isinstance(child, BodyJoint):
                        self.modules_without_complementaries.append(child)
                        self.body_joints += 1
                    else:
                        if child.complementary_limb not in self.modules_without_complementaries:
                            self.modules_without_complementaries.append(child)
                        self.limb_joints += 1
                        if isinstance(m, BodyJoint):
                            self.limbs += 1

    def reset_controllers(self):
        for module in self.modules:
            module.controller.reset()

    def get_next_action(self, action_array: np.array, delta_time: float) -> np.array:
        for i, module in enumerate(self.modules):
            action = module.controller.update(delta_time)
            action_array[0, i] = action

        return action_array

    def get_diversity_features(self) -> list:
        # Get number of body joints, limb joints and number of pair of limbs
        return [self.body_joints, self.limb_joints, self.limbs]

    def clean_up_genome(self, module_keys: list[str]):
        removed = 0
        for module in self.modules_without_complementaries:
            if module.name not in module_keys:
                self._remove_module(module)
                removed += 1
        if removed != 0:
            if removed == self.added:  # If all modules added since last eval failed the age is reset
                self.morph_age = self.prev_age + 1
                self.record.pop()
            self.generate_module_lists()
        self.added = 0

    def get_ordered_body_joints(self) -> list[Module]:
        root_children = self.root.children
        prev_module = None
        next_module = None
        for child in root_children:
            if child.connection_site == 3:
                prev_module = child
            if child.connection_site == 2:
                next_module = child
        prev_modules = []
        next_modules = []

        if prev_module != None:
            prev_module.DFS(prev_modules)
        if next_module != None:
            next_module.DFS(next_modules)

        return prev_modules[::-1] + [self.root] + next_modules
    
    def build_joint_table(self):
        body = self.get_ordered_body_joints()
        nums = []
        for m in body:
            joints = 1
            if len(m.children) > 1:
                for child in m.children:
                    if child.connection_site == 0:
                        joints += child.DFS_count(0)
            nums.append(joints)
        return nums
