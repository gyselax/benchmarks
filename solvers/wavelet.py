from benchopt import BaseSolver
from benchopt import safe_import_context

from benchmark_utils import WaveletCompressor

with safe_import_context() as import_ctx:
    import pywt  # noqa: F401


class Solver(BaseSolver):

    name = "wavelet"
    sampling_strategy = "run_once"
    requirements = ["numpy", "pip::pywavelets"]

    parameters = {
        "wavelet": ["db2"],
        "compression_ratio": [10],
        "level": [None],
        "mode": ["periodic"],
    }

    def set_objective(self, fields: dict):
        self.fields = fields

    def run(self, _):
        compressor = WaveletCompressor(
            wavelet=self.wavelet,
            compression_ratio=self.compression_ratio,
            level=self.level,
            mode=self.mode,
        )
        self.fields_rec = {
            name: compressor.compress_reconstruct(arr)
            for name, arr in self.fields.items()
        }

    def get_result(self) -> dict:
        return dict(fields_rec=self.fields_rec)
