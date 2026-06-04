"""Tiny JSON helpers to cache moment trajectories produced by ``prepare()``."""

import json
import pathlib


def dump_trajectory(traj: dict, path) -> None:
    """Write a moment trajectory (dict of lists) to ``path`` as JSON."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(traj, f)


def load_trajectory(path) -> dict:
    """Read a moment trajectory written by :func:`dump_trajectory`."""
    with open(path) as f:
        return json.load(f)
