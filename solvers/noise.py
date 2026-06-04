import numpy as np
from benchopt import BaseSolver


class Solver(BaseSolver):

    name = "noise"
    sampling_strategy = "run_once"
    requirements = ["numpy"]

    # Baseline "compression": additive Gaussian noise scaled by the field std,
    # to probe how sensitive the downstream evaluation is to perturbations.
    parameters = {
        "noise_level": [0.01],
    }

    def set_objective(self, fields: dict):
        self.fields = fields

    def run(self, _):
        rng = np.random.default_rng()
        level = float(self.noise_level)
        self.fields_rec = {
            name: arr + level * np.std(arr) * rng.standard_normal(
                arr.shape
            ).astype(arr.dtype)
            for name, arr in self.fields.items()
        }

    def get_result(self) -> dict:
        return dict(fields_rec=self.fields_rec)
