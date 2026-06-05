from benchopt import BaseObjective

from benchmark_utils.metrics import static_field_metrics, moment_conservation


class Objective(BaseObjective):

    name = "Physics Compression"
    url = "https://github.com/gyselax/benchmarks"
    min_benchopt_version = "1.8"
    # Compression is deterministic given parameters: a single pass is enough.
    sampling_strategy = "run_once"

    # Need H5 for processing simulations outputs
    requirements = ["h5py"]

    # Number of diagnostic steps run by the dynamic restart evaluation. Only
    # used by datasets that expose a restart function; ignored otherwise.
    parameters = {
        "restart_n_iter": [10],
    }

    def set_data(self, fields, moments_fn=None, restart_fn=None):
        # ``fields`` is the reference frame as a dict of named nD tensors.
        # ``moments_fn(fields) -> dict`` gives conserved quantities (optional).
        # ``restart_fn(fields_rec, n_iter) -> dict`` restarts the downstream
        # simulation and diffs it against the uncompressed restart (optional).
        self.fields = fields
        self.moments_fn = moments_fn
        self.restart_fn = restart_fn

    def get_objective(self) -> dict:
        return dict(fields=self.fields)

    def evaluate_result(self, fields_rec: dict) -> dict:
        # Static reconstruction quality, per field and averaged.
        results = static_field_metrics(self.fields, fields_rec)
        # benchopt minimises ``value``; use the mean MSE across fields.
        results["value"] = results["mse"]

        # Static moment-conservation error, when the dataset defines moments.
        if self.moments_fn is not None:
            results.update(moment_conservation(
                self.moments_fn(self.fields), self.moments_fn(fields_rec),
            ))

        # Dynamic evaluation: restart from the compressed frame and diff the
        # moment trajectory against the uncompressed restart.
        if self.restart_fn is not None:
            results.update(self.restart_fn(fields_rec, self.restart_n_iter))

        return results

    def get_one_result(self) -> dict:
        # Identity reconstruction (no compression) as a trivial reference.
        return dict(fields_rec=self.fields)
