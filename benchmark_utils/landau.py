"""Landau 2D2V moments, data generation and dynamic (restart) evaluation.

Moments are adapted from ``gysela-mini-app_io`` ``evaluate_compression.py`` and
extended with the electrostatic potential energy. The potential energy of a
single ``fdistribu`` frame is obtained from the density via an FFT Poisson
solve (periodic in x and y, matching the app's ``FFTPoissonSolver``); when a
solved potential is already available (the diagnostics store it), it is used
directly.

The dynamic evaluation restarts the compiled C++ mini-app from a frame and
collects per-diagnostic-step moments. ``landau_restart_trajectory`` is run once
for the compressed frame and once for the uncompressed frame; the objective
diffs the two trajectories.
"""

import os
import pathlib
import shutil
import subprocess
import tempfile

import numpy as np

# h5py and yaml are imported lazily inside the functions that use them, so that
# importing this module (e.g. for ``landau_moments``) needs only numpy.

EPS = 1e-12


def squeeze_species(f: np.ndarray) -> np.ndarray:
    """Return a single-species (Nx, Ny, Nvx, Nvy) view of ``fdistribu``."""
    f = np.asarray(f)
    if f.ndim == 5:
        return f[0]
    if f.ndim == 4:
        return f
    raise ValueError(f"Expected fdistribu rank 4 or 5, got shape {f.shape}.")


def _potential_energy_from_phi(phi, x, y) -> float:
    """Field energy 0.5 * int |grad phi|^2 (periodic centred differences)."""
    dx, dy = x[1] - x[0], y[1] - y[0]
    dphi_dx = (np.roll(phi, -1, 0) - np.roll(phi, 1, 0)) / (2.0 * dx)
    dphi_dy = (np.roll(phi, -1, 1) - np.roll(phi, 1, 1)) / (2.0 * dy)
    return float(0.5 * np.sum(dphi_dx**2 + dphi_dy**2) * dx * dy)


def _potential_energy_from_density(n, x, y) -> float:
    """Solve -lap phi = (n - mean n) by FFT, return the field energy."""
    nx, ny = n.shape
    dx, dy = x[1] - x[0], y[1] - y[0]
    rho = n - n.mean()
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    k2 = kx[:, None] ** 2 + ky[None, :] ** 2
    k2[0, 0] = 1.0
    phi_hat = np.fft.fft2(rho) / k2
    phi_hat[0, 0] = 0.0
    ex = np.real(np.fft.ifft2(1j * kx[:, None] * phi_hat))
    ey = np.real(np.fft.ifft2(1j * ky[None, :] * phi_hat))
    return float(0.5 * np.sum(ex**2 + ey**2) * dx * dy)


def landau_moments(f, x, y, vx, vy, phi=None) -> dict:
    """Mass, momentum, kinetic energy and potential energy of one species.

    ``phi`` (the solved electrostatic potential) is used for the potential
    energy when provided; otherwise it is reconstructed from the density via an
    FFT Poisson solve.
    """
    f = squeeze_species(f).astype(np.float64)
    x, y = np.asarray(x), np.asarray(y)
    vx, vy = np.asarray(vx), np.asarray(vy)
    dvx, dvy = vx[1] - vx[0], vy[1] - vy[0]
    dV = (x[1] - x[0]) * (y[1] - y[0]) * dvx * dvy

    f_vx = np.sum(f, axis=(0, 1, 3))
    f_vy = np.sum(f, axis=(0, 1, 2))

    mass = float(np.sum(f) * dV)
    momentum_x = float(np.dot(f_vx, vx) * dV)
    momentum_y = float(np.dot(f_vy, vy) * dV)
    momentum_norm = float(np.hypot(momentum_x, momentum_y))
    kinetic = float(0.5 * (np.dot(f_vx, vx**2) + np.dot(f_vy, vy**2)) * dV)

    density = np.sum(f, axis=(2, 3)) * dvx * dvy
    if phi is not None:
        potential = _potential_energy_from_phi(np.asarray(phi), x, y)
    else:
        potential = _potential_energy_from_density(density, x, y)

    return dict(mass=mass, momentum_x=momentum_x, momentum_y=momentum_y,
                momentum_norm=momentum_norm, kinetic_energy=kinetic,
                potential_energy=potential)


def read_mesh(mesh_h5) -> dict:
    """Read the (x, y, vx, vy) mesh from a GYSELALIBXX HDF5 file."""
    import h5py
    with h5py.File(mesh_h5, "r") as h5:
        return {k: h5[name][:] for k, name in
                (("x", "MeshX"), ("y", "MeshY"),
                 ("vx", "MeshVx"), ("vy", "MeshVy"))}


def read_landau_frame(h5_path, mesh_h5=None, dataset_name="fdistribu"):
    """Read one ``fdistribu`` tensor and its mesh."""
    import h5py
    with h5py.File(h5_path, "r") as h5:
        f = h5[dataset_name][:]
    return f, read_mesh(mesh_h5 if mesh_h5 is not None else h5_path)


def source_frame(source_h5, dataset_name="fdistribu") -> np.ndarray:
    """Read the uncompressed frame stored in ``source_h5``."""
    import h5py
    with h5py.File(source_h5, "r") as h5:
        return h5[dataset_name][:]


def landau_restart_available(binary, pdi_yaml, params_yaml) -> bool:
    """True when the compiled mini-app and its configs are all present."""
    return all(p is not None and os.path.exists(os.path.expanduser(str(p)))
               for p in (binary, pdi_yaml, params_yaml))


def _run_landau(binary, params_yaml, pdi_yaml, work_dir, *, nbiter,
                restart_file="none", nb_restart=0, n_ranks=4,
                mpi_launcher="mpirun"):
    """Run the mini-app in ``work_dir`` and return the list of diag files."""
    import yaml
    work_dir = pathlib.Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(open(os.path.expanduser(str(params_yaml))))
    cfg.setdefault("Input", {}).update(
        nb_restart=nb_restart, iter_offset=0,
        fdistribu_filename=str(restart_file))
    cfg.setdefault("Algorithm", {})["nbiter"] = int(nbiter)
    run_cfg = work_dir / "config.yaml"
    yaml.safe_dump(cfg, open(run_cfg, "w"))

    subprocess.run(
        [mpi_launcher, "-n", str(n_ranks),
         os.path.abspath(os.path.expanduser(str(binary))),
         str(run_cfg), os.path.abspath(os.path.expanduser(str(pdi_yaml)))],
        cwd=work_dir, check=True, env=os.environ.copy(),
    )
    return sorted(work_dir.glob("GYSELALIBXX_[0-9]*.h5"))


def generate_landau_frame(binary, params_yaml, pdi_yaml, out_dir, *,
                          n_iter, n_ranks=4, mpi_launcher="mpirun"):
    """Cold-start the mini-app for ``n_iter`` steps; cache, return the frame.

    Returns ``(frame_h5, initstate_h5)``. If the run directory already holds a
    completed run, it is reused.
    """
    out_dir = pathlib.Path(out_dir)
    diags = sorted(out_dir.glob("GYSELALIBXX_[0-9]*.h5"))
    if not diags:
        diags = _run_landau(
            binary, params_yaml, pdi_yaml, out_dir, nbiter=n_iter,
            nb_restart=0, n_ranks=n_ranks, mpi_launcher=mpi_launcher)
    initstate = out_dir / "GYSELALIBXX_initstate.h5"
    return diags[-1], (initstate if initstate.exists() else diags[-1])


def _read_diag_moments(diag_files, mesh, dataset_name="fdistribu") -> dict:
    """Per-diagnostic-step moment trajectory from a list of diag files."""
    import h5py
    traj = {k: [] for k in
            ("time", "mass", "momentum_x", "momentum_y", "momentum_norm",
             "kinetic_energy", "potential_energy")}
    for fp in diag_files:
        if "initstate" in str(fp):
            continue
        with h5py.File(fp, "r") as h5:
            f = h5[dataset_name][0]
            phi = h5["electrostatic_potential"][:] \
                if "electrostatic_potential" in h5 else None
            time = float(h5["time_saved"][()]) if "time_saved" in h5 else \
                len(traj["time"])
        m = landau_moments(f, phi=phi, **mesh)
        traj["time"].append(time)
        for k in m:
            traj[k].append(m[k])
    return traj


def landau_restart_trajectory(binary, params_yaml, pdi_yaml, source_h5, f,
                              mesh, *, n_iter, n_ranks=4,
                              mpi_launcher="mpirun", dataset_name="fdistribu"):
    """Restart from frame ``f`` for ``n_iter`` steps; moment trajectory."""
    import h5py
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        restart = tmp / "restart.h5"
        shutil.copy2(source_h5, restart)
        f = np.asarray(f)
        with h5py.File(restart, "r+") as h5:
            if h5[dataset_name].shape != f.shape:
                raise RuntimeError(
                    f"restart shape {h5[dataset_name].shape} != frame "
                    f"{f.shape}")
            h5[dataset_name][...] = f
        diags = _run_landau(
            binary, params_yaml, pdi_yaml, tmp / "run", nbiter=n_iter,
            nb_restart=1, restart_file=restart, n_ranks=n_ranks,
            mpi_launcher=mpi_launcher)
        return _read_diag_moments(diags, mesh, dataset_name)
