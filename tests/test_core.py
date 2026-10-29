"""Model files, jaw map, jaw muscle law and surrogate. No FLAME needed."""
import json
import math
import unittest

import numpy as np
import torch

from myoflame.jaw import JawCalibrationMap, JawMuscleLaw
from myoflame.model import DATA_DIR
from myoflame.surrogate import ActivationSurrogate
from myoflame.vocabulary import Vocabulary


def load_map():
    return JawCalibrationMap(dict(np.load(DATA_DIR / "jaw_calibration.npz")))


class ModelFiles(unittest.TestCase):
    def test_response_basis(self):
        grid = np.load(DATA_DIR / "muscle_responses.npz")
        R = grid["R_grid"]
        self.assertEqual(R.shape, (5, 5, 3 * 3931, 50))
        self.assertTrue(np.isfinite(R).all())
        self.assertTrue((np.diff(grid["jaw_open_bins"]) > 0).all())
        self.assertTrue((np.diff(grid["jaw_lateral_bins"]) > 0).all())
        obs = grid["flame_obs_idx"]
        self.assertEqual(len(np.unique(obs)), len(obs))
        self.assertTrue((obs >= 0).all() and (obs < 5023).all())

    def test_vocabulary(self):
        grid = np.load(DATA_DIR / "muscle_responses.npz")
        vocabulary = Vocabulary(grid["muscle_names"], grid["muscle_types"])
        self.assertEqual(len(vocabulary), 50)
        self.assertEqual(sorted(vocabulary.names[k] for k in vocabulary.jaw),
                         sorted(["Digastric_L", "Digastric_R", "Masseter_L", "Masseter_R",
                                 "Temporalis_L", "Temporalis_R"]))
        self.assertEqual(len(vocabulary.pairs), 25)
        for left, right in vocabulary.pairs:
            self.assertEqual(vocabulary.mirror_index(left), right)

    def test_expressions_and_action_units(self):
        data = np.load(DATA_DIR / "expressions.npz")
        self.assertEqual(data["expression"].shape, (218, 100))
        self.assertEqual(data["jaw_pose"].shape, (218, 3))
        self.assertEqual(len(set(data["label"].tolist())), 218)
        units = json.loads((DATA_DIR / "action_units.json").read_text())
        grid = np.load(DATA_DIR / "muscle_responses.npz")
        names = set(grid["muscle_names"].tolist())
        for entry in units.values():
            self.assertTrue(set(entry["muscles"]) <= names)


class JawMap(unittest.TestCase):
    def setUp(self):
        self.map = load_map()

    def test_interpolates_the_calibration_grid(self):
        qo, ql = np.meshgrid(self.map.open_bins, self.map.lateral_bins, indexing="ij")
        theta = self.map.physical_to_theta(torch.from_numpy(qo.ravel()), torch.from_numpy(ql.ravel()))
        np.testing.assert_allclose(theta.numpy(), self.map.theta_grid.reshape(-1, 3), atol=1e-12)

    def test_mirror_symmetry(self):
        q = torch.rand(64, dtype=torch.float64) * self.map.q_open_max
        lateral = (torch.rand(64, dtype=torch.float64) - 0.5) * 0.4
        a = self.map.physical_to_theta(q, lateral)
        b = self.map.physical_to_theta(q, -lateral)
        torch.testing.assert_close(b, a * torch.tensor([1.0, -1.0, -1.0], dtype=torch.float64), atol=1e-9, rtol=0)

    def test_projection_round_trip(self):
        generator = torch.Generator().manual_seed(0)
        q = torch.stack([torch.rand(200, generator=generator, dtype=torch.float64) * self.map.q_open_max,
                         (torch.rand(200, generator=generator, dtype=torch.float64) - 0.5) * math.radians(24)], dim=1)
        theta = self.map.physical_to_theta(q[:, 0], q[:, 1])
        projected, in_domain, residual = self.map.project(theta)
        torch.testing.assert_close(projected, q, atol=1e-9, rtol=0)
        self.assertTrue(bool(in_domain.all()))
        self.assertLess(float(residual.max()), 1e-9)

    def test_out_of_domain_is_flagged(self):
        theta = torch.tensor([[math.radians(35.0), 0.0, 0.0], [math.radians(-8.0), 0.0, 0.0],
                              [math.radians(10.0), 0.0, 0.0]], dtype=torch.float64)
        q, in_domain, _ = self.map.project(theta)
        self.assertEqual(in_domain.tolist(), [False, False, True])
        self.assertAlmostEqual(float(q[0, 0]), self.map.q_open_max, places=9)


class JawLaw(unittest.TestCase):
    def setUp(self):
        grid = np.load(DATA_DIR / "muscle_responses.npz")
        self.law = JawMuscleLaw(grid["muscle_names"], math.radians(20.0))

    def test_inverse_round_trip(self):
        q = torch.linspace(0.0, math.radians(20.0), 41, dtype=torch.float64)
        a = self.law.fill(torch.rand(41, 50, dtype=torch.float64), q)
        torch.testing.assert_close(self.law.q_open(a), q, atol=1e-12, rtol=0)
        self.assertTrue(bool((a[:, self.law.masseter + self.law.temporalis] == 0).all()))

    def test_closers(self):
        a = torch.zeros(3, 50)
        a[:, self.law.openers] = 0.7
        a[1, self.law.masseter] = 0.1
        a[2, self.law.temporalis] = 0.2
        opening = torch.rad2deg(self.law.q_open(a))
        self.assertAlmostEqual(float(opening[0]), 14.0, places=4)
        self.assertLess(float(opening[1]), float(opening[0]))
        self.assertLess(float(opening[2]), float(opening[0]))
        a[0, self.law.masseter] = 0.2               # a fifth of the masseter holds the jaw shut
        a[0, self.law.openers] = 1.0
        self.assertEqual(float(self.law.q_open(a[:1])), 0.0)


class Surrogate(unittest.TestCase):
    def test_loads_and_vanishes_at_neutral(self):
        model = ActivationSurrogate.from_checkpoint(DATA_DIR / "surrogate.pt")
        betas = torch.randn(4, 300) * 0.5
        a = model(betas, torch.zeros(4, 100), torch.zeros(4), torch.zeros(4))
        self.assertEqual(float(a.abs().max()), 0.0)
        psi = torch.randn(4, 100) * 0.5
        a = model(betas, psi, torch.full((4,), 0.1), torch.zeros(4))
        self.assertTrue(bool(((a >= 0) & (a <= 1)).all()))
        self.assertEqual(tuple(a.shape), (4, 50))


if __name__ == "__main__":
    unittest.main()
