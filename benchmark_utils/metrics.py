"""Static reconstruction metrics shared across datasets."""

import numpy as np

EPS = 1e-12


def mse(x_ref: np.ndarray, x_rec: np.ndarray) -> float:
    """Mean squared error between a reference and reconstructed field."""
    return float(((np.asarray(x_rec) - np.asarray(x_ref)) ** 2).mean())


def psnr(x_ref: np.ndarray, x_rec: np.ndarray) -> float:
    """Peak signal-to-noise ratio (dB), peak taken as max |x_ref|."""
    err = mse(x_ref, x_rec)
    peak = float(np.abs(x_ref).max())
    return float(10 * np.log10(peak**2 / max(err, EPS)))


def relative_error(ref: float, rec: float) -> float:
    """Relative absolute error |rec - ref| / max(|ref|, eps)."""
    return float(abs(rec - ref) / max(abs(ref), EPS))


def static_field_metrics(fields_ref: dict, fields_rec: dict) -> dict:
    """Per-field MSE and PSNR, plus the mean MSE/PSNR across fields.

    Returns a flat dict with keys ``<field>_mse``, ``<field>_psnr`` and the
    aggregates ``mse`` / ``psnr`` averaged over all fields.
    """
    out = {}
    mses, psnrs = [], []
    for name, ref in fields_ref.items():
        rec = fields_rec[name]
        m = mse(ref, rec)
        p = psnr(ref, rec)
        out[f"{name}_mse"] = m
        out[f"{name}_psnr"] = p
        mses.append(m)
        psnrs.append(p)
    out["mse"] = float(np.mean(mses))
    out["psnr"] = float(np.mean(psnrs))
    return out


def trajectory_diff(traj_comp: dict, traj_ref: dict, prefix="restart") -> dict:
    """Diff two per-step moment trajectories (compressed vs uncompressed).

    For each moment key shared by both trajectories (``time`` excluded) returns
    the final-step and max absolute error, and the full per-step error list
    ``<prefix>_<moment>_err_traj``.
    """
    out = {}
    for key in traj_comp:
        if key == "time" or key not in traj_ref:
            continue
        c = np.asarray(traj_comp[key], dtype=float)
        r = np.asarray(traj_ref[key], dtype=float)
        n = min(c.size, r.size)
        if n == 0:
            continue
        err = np.abs(c[:n] - r[:n])
        out[f"{prefix}_{key}_final_err"] = float(err[-1])
        out[f"{prefix}_{key}_max_err"] = float(err.max())
        out[f"{prefix}_{key}_err_traj"] = err.tolist()
    return out


def moment_conservation(moments_ref: dict, moments_rec: dict) -> dict:
    """Relative conservation error for each conserved moment.

    For every key in ``moments_ref`` returns ``<key>_cons_err`` as the relative
    error of the reconstructed moment w.r.t. the reference one.
    """
    out = {}
    for key, ref in moments_ref.items():
        out[f"{key}_cons_err"] = relative_error(
            float(ref), float(moments_rec[key]))
    return out
