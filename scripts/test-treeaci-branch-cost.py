#!/usr/bin/env python3
"""Check dimension adjustment against known coefficients."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("branch_cost", Path(__file__).with_name("run-treeaci-branch-cost.py"))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class BranchCostTests(unittest.TestCase):
    def test_case_matrix_is_complete_and_unique(self):
        for full, count in ((False, 120), (True, 432)):
            cases = list(runner.cases(full))
            self.assertEqual(len(cases), count)
            self.assertEqual(len(set(cases)), count)
            self.assertEqual({c[0] for c in cases}, {2, 3, 4})

    def test_dimension_fit_recovers_degree_residual(self):
        records = []
        for degree in (2, 3, 4):
            for size in (32, 128, 512, 2048):
                time = 11 + 0.25 * size + {2: 0, 3: 7, 4: 19}[degree]
                records.append({"config": {"scalar":"f64", "d":2, "mode":"cold", "batch":32},
                    "nodes":[{"node":"tree:0", "degree":degree, "local_elements":size,
                              "guard_ns":time*16, "guard_misses":16,
                              "guard_batches":{"points":16, "calls":1},
                              "frame_batches":{"points":0, "calls":0}}]})
        fits = runner.fit_nodes(records)
        self.assertEqual(len(fits), 1)
        for field, expected in (("intercept_ns", 11), ("beta_ns_per_element", 0.25),
                                ("degree3_residual_ns", 7), ("degree4_residual_ns", 19)):
            self.assertAlmostEqual(fits[0][field], expected, places=10)
        self.assertLess(fits[0]["rms_residual_ns"], 1e-10)

    def test_incomplete_degrees_are_not_fitted(self):
        self.assertEqual(runner.fit_nodes([]), [])
        self.assertEqual(runner.fit_nodes([{"error":"failed"}]), [])

    def test_paired_statistics_keep_regression_direction(self):
        self.assertEqual(runner.interval([1.25]*5), [1.25, 1.25])
        self.assertEqual(runner.relative_mad([10,10,10,11,9]), 0)


if __name__ == "__main__":
    unittest.main()
