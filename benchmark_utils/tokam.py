"""Tokam2D moments, data generation and dynamic (restart) evaluation.

Uses the installable ``tokam2d`` package directly (in-process), so generation
and restart are plain Python calls — no subprocess, no CLI flags, no HDF5
round-trip. ``tokam_moments`` gives the single-frame conserved quantities for
the static check; ``run_tokam`` runs the simulation (from the analytic IC or
restarting from given fields) and ``tokam_trajectory`` turns its output into a
per-diagnostic-step moment trajectory for the dynamic comparison.
"""
import numpy as np


def tokam_moments(density, potential, x, y) -> dict:
    """Single-frame conserved quantities for a tokam2d state."""
    dV = (x[1] - x[0]) * (y[1] - y[0])
    mass = float(np.sum(density) * dV)
    thermal = float(0.5 * np.sum(density**2) * dV)
    dphi_dy, dphi_dx = np.gradient(potential, y, x)
    kinetic = float(0.5 * np.sum(dphi_dx**2 + dphi_dy**2) * dV)
    return dict(mass=mass, thermal_energy=thermal, kinetic_energy=kinetic,
                energy=thermal + kinetic)


def tokam_trajectory(out: dict) -> dict:
    """Per-diagnostic-step moment trajectory from a ``run_tokam`` output."""
    density = np.asarray(out["fields"]["density"])
    potential = np.asarray(out["fields"]["potential"])
    x, y = np.asarray(out["x"]), np.asarray(out["y"])
    time = np.asarray(out["time"]).tolist()
    traj = {k: [] for k in ("time", "mass", "thermal_energy",
                            "kinetic_energy", "energy")}
    for t in range(density.shape[0]):
        m = tokam_moments(density[t], potential[t], x, y)
        traj["time"].append(time[t] if t < len(time) else float(t))
        for k in m:
            traj[k].append(m[k])
    return traj
