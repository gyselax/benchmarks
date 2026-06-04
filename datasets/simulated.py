import numpy as np
from benchopt import BaseDataset

from benchmark_utils import tokam_moments, landau_moments


class Dataset(BaseDataset):

    name = "Simulated"
    # Synthetic frames with the same shapes/moments as the real cases, so the
    # benchmark runs end to end without external data or binaries.
    parameters = {
        "case": ["tokam2d", "landau2X2V"],
    }
    requirements = ["numpy"]

    def get_data(self) -> dict:
        rng = np.random.default_rng(0)

        if self.case == "tokam2d":
            nx, ny = 64, 64
            x = np.linspace(0.0, 2 * np.pi, nx)
            y = np.linspace(0.0, 2 * np.pi, ny)
            xx, yy = np.meshgrid(x, y, indexing="ij")
            density = (1.0 + 0.1 * np.sin(xx) * np.cos(yy)).astype(np.float32)
            potential = (0.05 * np.sin(2 * xx + yy)).astype(np.float32)
            fields = {"density": density, "potential": potential}

            def moments_fn(f):
                return tokam_moments(f["density"], f["potential"], x, y)

            return dict(fields=fields, moments_fn=moments_fn, restart_fn=None)

        # landau2X2V: single-species fdistribu[species, x, y, vx, vy].
        nx = ny = 16
        nvx = nvy = 17
        x = np.linspace(0.0, 2 * np.pi, nx)
        y = np.linspace(0.0, 2 * np.pi, ny)
        vx = np.linspace(-6.0, 6.0, nvx)
        vy = np.linspace(-6.0, 6.0, nvy)
        maxwellian = np.exp(-0.5 * (vx[:, None] ** 2 + vy[None, :] ** 2))
        perturb = 1.0 + 0.05 * np.sin(x)[:, None, None, None]
        f = (perturb * maxwellian[None, None, :, :]).astype(np.float32)
        f = np.broadcast_to(f, (nx, ny, nvx, nvy)).copy()
        f += (1e-3 * rng.standard_normal(f.shape)).astype(np.float32)
        fields = {"fdistribu": f[None]}  # leading species axis

        mesh = dict(x=x, y=y, vx=vx, vy=vy)

        def moments_fn(fr):
            return landau_moments(fr["fdistribu"], **mesh)

        return dict(fields=fields, moments_fn=moments_fn, restart_fn=None)
