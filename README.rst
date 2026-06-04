Benchmark for compression of physics simulation frames
=======================================================

|Build Status|

A `benchopt <https://benchopt.github.io/>`_ benchmark for **lossy compression of
physics simulation frames**. It evaluates compression methods both *statically*
(reconstruction quality of a single frame) and *dynamically* (using the
compressed frame as a restart point for the downstream simulation).

It covers two physics use cases under one generic interface:

- **Tokam2D** — a 2D turbulence code (pure Python). The frame is the pair of
  fields ``{density, potential}``; the dynamic restart re-runs the simulation.
- **Landau2X2V** — the GYSELA 2D2V Landau-damping mini-app (C++/MPI). The frame
  is the 5D distribution ``fdistribu[species, x, y, vx, vy]``; the dynamic
  restart runs the compiled mini-app.

The real datasets **generate their own data** in ``Dataset.prepare()`` (benchopt
1.9): they run the simulation from the analytic initial condition for
``n_iter_init`` steps, and precompute the *uncompressed* restart reference
trajectory over ``restart_n_iter_ref`` steps. Both are cached under the
benchopt data folder (``get_data_path``) and reused on subsequent runs. Prepare
explicitly with:

.. code-block:: bash

   benchopt prepare benchmark_gysela -d Tokam2D
   # or, as part of a run:
   benchopt run benchmark_gysela --prepare ...

At evaluation, the dynamic restart runs the simulation for ``restart_n_iter``
steps from the *compressed* frame and reports the difference against the
precomputed uncompressed reference (``restart_n_iter`` must be
``<= restart_n_iter_ref``).

Install & run
-------------

.. code-block:: bash

   pip install benchopt
   benchopt run benchmark_gysela

To run only the dependency-free smoke test (no external data needed):

.. code-block:: bash

   benchopt run benchmark_gysela -d Simulated -s noise -s wavelet

To select a real dataset and method:

.. code-block:: bash

   benchopt run benchmark_gysela -d Tokam2D -s wavelet[compression_ratio=10]

Configuration is done entirely through **benchopt parameters** — there is no
separate config file. Pass them inline or via a run config (see
``example_config.yml``):

.. code-block:: bash

   benchopt run benchmark_gysela --config example_config.yml

Datasets
--------

================  ====================================  =======================
Dataset           Fields exposed                        Dynamic restart
================  ====================================  =======================
``Simulated``     synthetic tokam2d / landau2X2V         none
``Tokam2D``       ``density``, ``potential`` (2D)        Python tokam2d driver
``Landau2X2V``    ``fdistribu`` (5D)                     C++/MPI mini-app
================  ====================================  =======================

Key dataset parameters: ``n_iter_init`` (steps used to generate the frame),
``restart_n_iter_ref`` (reference horizon), ``base_config`` (input YAML — an
existing path, or a file name resolved against the shipped config folder
``get_data_path("<dataset>")/configs/``), and for Landau ``n_ranks`` and
``launcher`` (how to run the mini-app). ``Tokam2D`` uses the installable
``tokam2d`` package directly (in-process, no subprocess); ``Landau2X2V`` runs
the compiled C++/MPI mini-app inside docker. If the package or image is not
available the real dataset raises an actionable error; ``Simulated`` runs from
synthetic data.

Running the Landau mini-app (docker)
------------------------------------

The mini-app's runtime (liblapacke, MPI, …) exists only inside the
``gyselalibxx`` docker image, so it cannot run on a bare host. The binary is
baked into an image (no host binary to mount), and the dataset stays agnostic
to *how* it is launched: it invokes the ``launcher`` parameter (an executable,
no spaces) with a minimal contract::

   launcher <n_ranks> <config> <work_dir>

**1. Build the image once** from a compiled ``gysela-mini-app_io`` checkout
(the ``Dockerfile`` copies the statically-linked binary + PDI config into the
gyselalibxx runtime image)::

   docker build -f benchmark_gysela/Dockerfile -t gysela-compression gysela-mini-app_io

**2. Point ``launcher`` at the shipped ``landau_docker_launch.sh``** (an
absolute path). It wraps each call in ``docker run`` against that image,
mounting only the work dir (run config + outputs) at its same host path, as the
current host user. Override with ``GYSELA_IMAGE`` / ``GYSELA_BIN`` /
``GYSELA_PDI``. Example run config::

   dataset:
     - Landau2X2V[n_ranks=4,launcher=/abs/benchmark_gysela/landau_docker_launch.sh]

Alternatively, run benchopt **inside** the baked image and leave ``launcher``
empty — the dataset then runs ``mpirun`` directly against the baked
``binary``/``pdi`` (defaults ``/opt/gysela/...``).

Note: benchopt's ``name[param=value]`` syntax cannot contain spaces in a value;
that is why ``launcher`` is a single script path rather than a full command.

Dataset contract
----------------

Every dataset's ``get_data()`` returns:

- ``fields``: ``dict[str, ndarray]`` — the reference frame to compress.
- ``moments_fn(fields) -> dict`` *(optional)* — conserved quantities for the
  static moment-conservation check.
- ``restart_fn(fields_rec, n_iter) -> dict`` *(optional)* — restarts the
  downstream simulation for ``n_iter`` steps from the reconstructed frame and
  returns the moment-trajectory difference w.r.t. the uncompressed restart.

Solvers compress each field by name, so they are independent of the use case.

``benchmark_utils`` import convention
-------------------------------------

Helpers are imported from their **submodule** (``from benchmark_utils.tokam
import ...``), never from the package root, and ``benchmark_utils/__init__.py``
is intentionally empty. Submodules are organised **by topic** (``metrics``,
``tokam``, ``landau``, ``wavelet``, ``storage``), and each *component*
(dataset/solver) declares the requirements of the helpers it imports —
dependency isolation is at the component level, not per submodule. So
``landau.py`` holds all things Landau (moments + HDF5/MPI) and imports
``h5py``/``yaml`` at its top; any component importing it (``Landau2X2V``, and
``Simulated`` for the synthetic landau moments) lists those in its
``requirements``. Imports are **not** hidden inside functions, so a module that
needs ``h5py`` raises at load when it is missing — which is how benchopt detects
the owning component as not-installed.

Solvers
-------

- ``noise`` — adds Gaussian noise scaled by the field std (a baseline that
  probes downstream sensitivity at various noise levels).
- ``wavelet`` — nD wavelet decomposition, keeps the largest coefficients
  (fraction set by ``compression_ratio``), float16 quantisation.

Metrics
-------

- Per-field ``<field>_mse`` / ``<field>_psnr`` plus averaged ``mse`` / ``psnr``
  (the objective ``value`` is the mean MSE).
- ``<moment>_cons_err`` — relative conservation error of each moment. For
  Landau this includes ``potential_energy`` (solved from the reconstructed
  density via FFT Poisson).
- Dynamic restart, when ``restart_fn`` is available:
  ``restart_<moment>_final_err`` / ``restart_<moment>_max_err`` and the full
  per-step error list ``restart_<moment>_err_traj``.

.. |Build Status| image:: https://github.com/gyselax/benchmark_gysela/workflows/Tests/badge.svg
   :target: https://github.com/gyselax/benchmark_gysela/actions
