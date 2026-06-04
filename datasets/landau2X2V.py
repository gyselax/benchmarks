import pathlib

from benchopt import BaseDataset
from benchopt import safe_import_context


from benchmark_utils.landau import (
    landau_moments, read_mesh, source_frame, landau_restart_available,
    generate_landau_frame, landau_restart_trajectory
)
from benchmark_utils.metrics import trajectory_diff
from benchmark_utils.storage import dump_trajectory, load_trajectory

with safe_import_context() as import_ctx:
    import numpy as np
    from benchopt.config import get_data_path

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent
# Default locations follow the current repo layout; override via the benchopt
# run config (see example_config.yml).
_DEFAULT_APP = str(_PROJECT_ROOT.parent / "gysela-mini-app_io")


def _app_paths(app_root):
    root = pathlib.Path(app_root)
    return (
        root / "build" / "apps" / "compression" / "compression_app",
        root / "apps" / "compression" / "params.yaml",
        root / "apps" / "compression" / "pdi_out.yaml",
    )


class Dataset(BaseDataset):

    name = "Landau2X2V"
    # Paths and generation settings are benchopt parameters: set them from the
    # run config rather than a separate config file.
    parameters = {
        "dtype": ["float32"],
        "n_iter_init": [10],
        # Horizon of the precomputed uncompressed-restart reference. Objective
        # restart_n_iter must be <= this value.
        "restart_n_iter_ref": [10],
        "n_ranks": [4],
        "landau_app_root": [_DEFAULT_APP],
        "mpi_launcher": ["mpirun"],
    }
    requirements = ["h5py"]
    prepare_cache_ignore = ("dtype",)

    def _data_dir(self):
        return get_data_path(
            f"landau2X2V_n{self.n_iter_init}_ref{self.restart_n_iter_ref}")

    def _ensure_prepared(self):
        """Cold-start the app and precompute the restart reference (cached)."""
        binary, params_yaml, pdi_yaml = _app_paths(self.landau_app_root)
        if not landau_restart_available(binary, params_yaml, pdi_yaml):
            raise RuntimeError(
                "Landau2X2V needs the compiled mini-app. Expected the binary "
                f"at '{binary}' plus params.yaml/pdi_out.yaml. Build "
                "gysela-mini-app_io (BUILD_COMPRESSION_APP=ON) and/or set "
                "'landau_app_root' in the benchopt run config."
            )
        data_dir = self._data_dir()
        frame_h5, mesh_h5 = generate_landau_frame(
            binary, params_yaml, pdi_yaml, data_dir,
            n_iter=self.n_iter_init, n_ranks=self.n_ranks,
            mpi_launcher=self.mpi_launcher,
        )
        ref_path = data_dir / "reference_restart.json"
        if not ref_path.exists():
            mesh = read_mesh(mesh_h5)
            ref = landau_restart_trajectory(
                binary, params_yaml, pdi_yaml, frame_h5,
                source_frame(frame_h5), mesh, n_iter=self.restart_n_iter_ref,
                n_ranks=self.n_ranks, mpi_launcher=self.mpi_launcher)
            dump_trajectory(ref, ref_path)
        return frame_h5, mesh_h5, ref_path

    def prepare(self):
        self._ensure_prepared()

    def get_data(self) -> dict:
        binary, params_yaml, pdi_yaml = _app_paths(self.landau_app_root)
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
                binary, params_yaml, pdi_yaml, frame_h5,
                np.asarray(fr["fdistribu"]), mesh, n_iter=min(n_iter, n_ref),
                n_ranks=self.n_ranks, mpi_launcher=self.mpi_launcher)
            return trajectory_diff(comp, reference)

        return dict(
            fields=fields, moments_fn=moments_fn, restart_fn=restart_fn)
