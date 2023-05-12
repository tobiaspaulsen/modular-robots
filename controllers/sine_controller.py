import math
import matplotlib.pyplot as plt
import numpy as np
import random
import config
from controllers.controller import Controller


class SineController(Controller):
    # TODO: Tune this
    allowable_amp = (0.0, 3.0)
    # allowable_freq = (0.0, 2.5)
    allowable_phase = (-np.inf, np.inf)
    allowable_offset = (-1, 1)

    def __init__(self, node_id: str, parent: Controller):
        super().__init__(node_id, parent)
        self.state = 0.0
        self.amp = random.uniform(0.5, 2)  # Check this
        self.freq = 1.0
        self.phase = random.uniform(-1, 1)
        self.offset = random.uniform(*SineController.allowable_offset)

    def __str__(self):
        string = "Amp:".ljust(10, " ") + f"{round(self.amp, 2)}\n"
        string += "Freq:".ljust(10, " ") + f"{round(self.freq, 2)}\n"
        string += "Phase:".ljust(10, " ") + f"{round(self.phase, 2)}\n"
        string += "Offset:".ljust(10, " ") + f"{round(self.offset, 2)}\n"
        return string

    def update(self, delta_time: float) -> np.float32:
        self.state += delta_time
        out = self.amp * math.sin(2 * np.pi * self.freq * self.state + self.phase) + self.offset
        return np.clip(out, config.MIN_CONTROLLER_OUTPUT, config.MAX_CONTROLLER_OUTPUT)

    def reset(self):
        self.state = 0

    def mutate(self, mutation_rate: float, mutation_sigma: float):
        # Mutation rate is divided by 3 because of the 3 possible mutations
        mutation_rate = mutation_rate / 3
        if random.uniform(0, 1) < mutation_rate:
            self.amp = np.clip(random.gauss(self.amp, mutation_sigma), *SineController.allowable_amp)
        if random.uniform(0, 1) < mutation_rate:
            self.phase = np.clip(random.gauss(self.phase, mutation_sigma), *SineController.allowable_phase)
        if random.uniform(0, 1) < mutation_rate:
            self.offset = np.clip(random.gauss(self.offset, mutation_sigma), *SineController.allowable_offset)


