import math
import numpy as np
import random
import config
from controllers.controller import Controller


class CoupledOscillator(Controller):
    allowable_amp = (0.0, 2.0)
    allowable_phase_offset = (-np.pi, np.pi)
    allowable_offset = (-1.0, 1.0)

    def __init__(self, node_id: str, parent: Controller, init: bool = False):
        super().__init__(node_id, parent)
        if init or parent is None:
            self.amp = random.uniform(0.5, 2)
            self.phase_offset = random.uniform(*CoupledOscillator.allowable_phase_offset)
            self.offset = random.uniform(*CoupledOscillator.allowable_offset)
        else:
            self.amp = parent.amp
            self.phase_offset = parent.phase_offset
            self.offset = parent.offset
           
        self.time_state = 0.0
        self.phase_state = 0
        self.freq = 4

    def update(self, delta_time: float) -> np.float32:
        self.time_state += delta_time
        if self.parent is not None:
            self.phase_state = self.parent.phase_state + self.phase_offset
        out = self.amp * math.sin(self.freq * self.time_state + self.phase_state) + self.offset
        return np.clip(out, config.MIN_CONTROLLER_OUTPUT, config.MAX_CONTROLLER_OUTPUT)

    def reset(self):
        self.phase_state = 0
        self.time_state = 0

    def mutate(self, mutation_rate: float, mutation_sigma: float):
        if random.uniform(0, 1) < mutation_rate:
            scaled_sigma = mutation_sigma * (CoupledOscillator.allowable_amp[1] - CoupledOscillator.allowable_amp[0])
            self.amp = np.clip(random.gauss(self.amp, scaled_sigma), *CoupledOscillator.allowable_amp)
        if random.uniform(0, 1) < mutation_rate:
            scaled_sigma = mutation_sigma * (CoupledOscillator.allowable_phase_offset[1] - CoupledOscillator.allowable_phase_offset[0])
            self.phase_offset = random.gauss(self.phase_offset, scaled_sigma)
            if self.phase_offset < self.allowable_phase_offset[0]:
                diff = self.phase_offset - self.allowable_phase_offset[0]
                self.phase_offset = self.allowable_phase_offset[1] + diff
            elif self.phase_offset > self.allowable_phase_offset[1]:
                diff = self.phase_offset - self.allowable_phase_offset[1]
                self.phase_offset = self.allowable_phase_offset[0] + diff
        if random.uniform(0, 1) < mutation_rate:
            scaled_sigma = mutation_sigma * (CoupledOscillator.allowable_offset[1] - CoupledOscillator.allowable_offset[0])
            self.offset = np.clip(random.gauss(self.offset, scaled_sigma), *CoupledOscillator.allowable_offset)

    def __str__(self):
        string = "Amp:".ljust(10, " ") + f"{round(self.amp, 2)}\n"
        string += "Freq:".ljust(10, " ") + f"{round(self.freq, 2)}\n"
        string += "Phase offset:".ljust(10, " ") + f"{round(self.phase_offset, 2)}\n"
        string += "Offset:".ljust(10, " ") + f"{round(self.offset, 2)}\n"
        return string
