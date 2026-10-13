"""Audit 1, C1: the result files of part 2 were computed by the code in the repository. Not run by tests/mutate_mpc.py
(every mutation of a run file would fail it trivially).

    python -m unittest discover -s tests
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_mpc as rm  # noqa: E402


class ResultFilesMatchTheCode(unittest.TestCase):
    def test_fingerprint_and_no_cache(self):
        fp = rm.code_fingerprint()
        for name in ("mpc_seeds.json", "mpc_metrics.json"):
            pv = json.loads((ROOT / "outputs" / name).read_text()).get("provenance")
            self.assertIsNotNone(pv, f"{name}: provenance not recorded")
            self.assertEqual(pv["code_sha256_16"], fp, f"{name} was computed with other code or settings: re-run it")
            self.assertEqual(pv["runs_from_cache"], 0, f"{name}: runs taken from a cache")


if __name__ == "__main__":
    unittest.main()
