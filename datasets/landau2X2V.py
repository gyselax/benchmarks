import os
import pathlib

import numpy as np
from benchopt import BaseDataset
from benchopt.config import get_data_path

from benchmark_utils.landau import (
    landau_moments, read_mesh, source_frame, generate_landau_frame,
    landau_restart_trajectory, DEFAULT_BINARY, DEFAULT_PDI,
)
from benchmark_utils.metrics import trajectory_diff
from benchmark_utils.storage import dump_trajectory, load_trajectory


class Dataset(BaseDataset):

    name = "Landau2X2V"
    # Settings are benchopt parameters set from the run config. The mini-app is
    # run via `launcher` (see landau_docker_launch.sh) which owns the binary +
    # PDI config (baked into a docker image); leave it empty only when benchopt
    # itself runs inside that image, where `binary`/`pdi` default to the baked
    # paths. `base_config` is the GYSELA params YAML, resolved against the
    # shipped config folder get_data_path("landau2X2V")/configs/.
    parameters = {
        "dtype": ["float32"],
        "n_iter_init": [10],
        # Horizon of the precomputed uncompressed-restart reference. Objective
        # restart_n_iter must be <= this value.
        "restart_n_iter_ref": [10],
        "n_ranks": [4],
        "base_config": ["params.yaml"],
        # How to run the mini-app. Defaults to the shipped docker wrapper
        # (resolved relative to the benchmark), which runs it in the baked
        # image; set to "" to run mpirun directly (benchopt inside that image).
        "launcher": ["landau_docker_launch.sh"],
        "binary": [DEFAULT_BINARY],
        "pdi": [DEFAULT_PDI],
    }
    # tiny 16x16x17x17 config + a couple of steps to stay fast. Inherits the
    # docker launcher default, so `benchopt test` runs the real mini-app when
    # the gysela-compression image is available.
    test_parameters = {
        "n_iter_init": [1],
        "restart_n_iter_ref": [1],
        "n_ranks": [2],
        "base_config": ["params.yaml"],
    }
    requirements = ["h5py", "pyyaml"]
    prepare_cache_ignore = ("dtype",)

    def _data_dir(self):
        return get_data_path(
            f"landau2X2V_n{self.n_iter_init}_ref{self.restart_n_iter_ref}")

    def _resolve_config(self):
        """Resolve base_config: use it if it exists, else look it up under
        the shipped config folder get_data_path("landau2X2V")/configs/."""
        given = pathlib.Path(self.base_config).expanduser()
        if given.exists():
            return given
        candidate = get_data_path("landau2X2V") / "configs" / self.base_config
        if candidate.exists():
            return candidate
        raise RuntimeError(
            f"Landau2X2V base_config '{self.base_config}' not found, neither "
            f"as a path nor under '{candidate.parent}'."
        )

    def _resolve_launcher(self):
        """Resolve launcher to an absolute path: empty (direct mpirun), a name
        resolved relative to the benchmark (where the script ships), or an
        existing path. Always absolute, since it is run with cwd=work_dir."""
        if not self.launcher:
            return ""
        candidate = pathlib.Path(__file__).resolve().parent.parent \
            / self.launcher
        if candidate.exists():
            return str(candidate)
        given = pathlib.Path(self.launcher).expanduser()
        if given.exists():
            return str(given.resolve())
        raise RuntimeError(
            f"Landau2X2V launcher '{self.launcher}' not found, neither under "
            f"'{candidate.parent}' nor as a path."
        )

    def _run_kwargs(self, config):
        launcher = self._resolve_launcher()
        if not launcher and not os.path.exists(self.binary):
            raise RuntimeError(
                "Landau2X2V cannot run the mini-app: set 'launcher' to "
                "landau_docker_launch.sh (host, see example_config.yml), or "
                "run benchopt inside the baked image where 'binary' exists."
            )
        return dict(launcher=launcher, binary=self.binary, pdi=self.pdi)

    def _ensure_prepared(self):
        """Cold-start the app and precompute the restart reference (cached)."""
        config = self._resolve_config()
        run_kwargs = self._run_kwargs(config)
        data_dir = self._data_dir()
        frame_h5, mesh_h5 = generate_landau_frame(
            config, data_dir, n_iter=self.n_iter_init, n_ranks=self.n_ranks,
            **run_kwargs)
        ref_path = data_dir / "reference_restart.json"
        if not ref_path.exists():
            mesh = read_mesh(mesh_h5)
            ref = landau_restart_trajectory(
                config, frame_h5, source_frame(frame_h5), mesh,
                n_iter=self.restart_n_iter_ref, n_ranks=self.n_ranks,
                **run_kwargs)
            dump_trajectory(ref, ref_path)
        return frame_h5, mesh_h5, ref_path

    def prepare(self):
        self._ensure_prepared()

    def get_data(self) -> dict:
        config = self._resolve_config()
        run_kwargs = self._run_kwargs(config)
        frame_h5, mesh_h5, ref_path = self._ensure_prepared()
        mesh = read_mesh(mesh_h5)
        fields = {"fdistribu": source_frame(frame_h5).astype(self.dtype)}
        reference = load_trajectory(ref_path)
        n_ref = self.restart_n_iter_ref

        def moments_fn(fr):
            return landau_moments(fr["fdistribu"], **mesh)

        def restart_fn(fr, n_iter):
            if n_iter > n_ref:
                print(f"[Landau2X2V] restart_n_iter={n_iter} exceeds the "
                      f"reference horizon {n_ref}; comparing over {n_ref} "
                      "steps. Increase restart_n_iter_ref to extend it.")
            comp = landau_restart_trajectory(
                config, frame_h5, np.asarray(fr["fdistribu"]), mesh,
                n_iter=min(n_iter, n_ref), n_ranks=self.n_ranks, **run_kwargs)
            return trajectory_diff(comp, reference)

        return dict(
            fields=fields, moments_fn=moments_fn, restart_fn=restart_fn)
