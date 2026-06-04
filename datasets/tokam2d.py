
from benchopt import BaseDataset
from benchopt.config import get_data_path

import pathlib
import numpy as np

from tokam2d import run_simulation

from benchmark_utils.tokam import tokam_moments, tokam_trajectory
from benchmark_utils.storage import dump_trajectory, load_trajectory
from benchmark_utils.metrics import trajectory_diff


class Dataset(BaseDataset):

    name = "Tokam2D"
    # Settings are benchopt parameters: set them from the run config rather
    # than a separate config file. The tokam2d simulation code is a pip
    # dependency (the installable package), not a path.
    parameters = {
        "dtype": ["float32"],
        "n_iter_init": [50],
        # Horizon of the precomputed uncompressed-restart reference. Objective
        # restart_n_iter must be <= this value.
        "restart_n_iter_ref": [50],
        # Input config: an existing path, or a file name resolved against the
        # shipped config folder get_data_path("tokam2d")/configs/.
        "base_config": ["input_SOL_interchange_driftwave.yaml"],
    }
    # `benchopt test` uses a tiny 16x16 config and 2 steps to stay fast.
    test_parameters = {
        "n_iter_init": [2],
        "restart_n_iter_ref": [2],
        "base_config": ["test_small.yaml"],
    }
    requirements = [
        "numpy",
        "pip::tokam2d[cpu] @ git+ssh://git@github.com/gyselax/tokam2d.git",
    ]
    prepare_cache_ignore = ("dtype",)

    def _data_dir(self):
        return get_data_path(
            f"tokam2d_n{self.n_iter_init}_ref{self.restart_n_iter_ref}")

    def _resolve_config(self):
        """Resolve base_config: use it if it exists, else look it up under
        the shipped config folder get_data_path("tokam2d")/configs/."""
        given = pathlib.Path(self.base_config).expanduser()
        if given.exists():
            return given
        candidate = get_data_path("tokam2d") / "configs" / self.base_config
        if candidate.exists():
            return candidate
        raise RuntimeError(
            f"Tokam2D base_config '{self.base_config}' not found, neither as "
            f"a path nor under '{candidate.parent}'. Provide an existing "
            "config path or drop the file in that folder."
        )

    def _ensure_prepared(self):
        """Generate the frame and the uncompressed restart reference."""
        config = self._resolve_config()
        data_dir = self._data_dir()
        frame_npz = data_dir / "frame.npz"
        if not frame_npz.exists():
            out = run_simulation(config, n_iter=self.n_iter_init)
            frame_npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                frame_npz, x=out["x"], y=out["y"],
                density=out["final_fields"]["density"],
                potential=out["final_fields"]["potential"],
            )
        ref_path = data_dir / "reference_restart.json"
        if not ref_path.exists():
            f = np.load(frame_npz)
            out = run_simulation(
                config, n_iter=self.restart_n_iter_ref,
                initial_fields={"density": f["density"],
                                "potential": f["potential"]},
            )
            dump_trajectory(tokam_trajectory(out), ref_path)
        return frame_npz, ref_path

    def prepare(self):
        self._ensure_prepared()

    def get_data(self) -> dict:
        frame_npz, ref_path = self._ensure_prepared()
        config = self._resolve_config()
        f = np.load(frame_npz)
        x, y = f["x"], f["y"]
        fields = {"density": f["density"].astype(self.dtype),
                  "potential": f["potential"].astype(self.dtype)}
        reference = load_trajectory(ref_path)
        n_ref = self.restart_n_iter_ref

        def moments_fn(fr):
            return tokam_moments(fr["density"], fr["potential"], x, y)

        def restart_fn(fr, n_iter):
            if n_iter > n_ref:
                print(f"[Tokam2D] restart_n_iter={n_iter} exceeds the "
                      f"reference horizon {n_ref}; comparing over {n_ref} "
                      "steps. Increase restart_n_iter_ref to extend it.")
            out = run_simulation(
                config, n_iter=min(n_iter, n_ref), initial_fields=fr
            )
            return trajectory_diff(tokam_trajectory(out), reference)

        return dict(
            fields=fields, moments_fn=moments_fn, restart_fn=restart_fn)
