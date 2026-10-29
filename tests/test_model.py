"""The full model. Needs FLAME 2020 in model/FLAME2020/ and skips otherwise."""
import math
import unittest
import warnings
from pathlib import Path

import torch

from myoflame import MyoFLAME, load_expressions

FLAME = Path(__file__).resolve().parents[1] / "model" / "FLAME2020" / "generic_model.pkl"
needs_flame = unittest.skipUnless(FLAME.is_file(), "FLAME 2020 not found in model/FLAME2020/")


@needs_flame
class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.model = MyoFLAME(FLAME)
        cls.expressions = load_expressions()
        labels = cls.expressions["label"].tolist()
        cls.index = [labels.index(x) for x in ("12", "1+2", "27", "18")]

    def batch(self):
        e = self.expressions
        return (torch.from_numpy(e["expression"][self.index]), torch.from_numpy(e["jaw_pose"][self.index]))

    def test_neutral_is_flame(self):
        betas = torch.randn(3, 300) * 0.5
        pose = torch.randn(3, 3) * 0.1
        out = self.model(betas=betas, global_orient=pose)
        flame = self.model.flame_forward(betas=betas, global_orient=pose)
        self.assertEqual(float((out.vertices - flame).abs().max()), 0.0)
        self.assertEqual(float(out.activations.abs().max()), 0.0)

    def test_output_contract(self):
        expression, jaw = self.batch()
        out = self.model(expression=expression, jaw_pose=jaw)
        self.assertEqual(tuple(out.vertices.shape), (4, 5023, 3))
        a = out.activations
        self.assertTrue(bool(((a >= 0) & (a <= 1)).all()))
        law = self.model.jaw_law
        torch.testing.assert_close(a[:, law.openers[0]], (out.jaw_open / law.q_open_max).clamp(0, 1))
        self.assertEqual(float(a[:, law.masseter + law.temporalis].abs().max()), 0.0)
        self.assertGreater(math.degrees(float(out.jaw_open[2])), 12.0)        # AU27 opens the jaw

    def test_batch_equals_single(self):
        expression, jaw = self.batch()
        out = self.model(expression=expression, jaw_pose=jaw)
        for b in range(len(expression)):
            single = self.model(expression=expression[b], jaw_pose=jaw[b])
            torch.testing.assert_close(single.vertices[0], out.vertices[b], atol=1e-6, rtol=0)
            torch.testing.assert_close(single.activations[0], out.activations[b], atol=1e-6, rtol=0)

    def test_truncated_parameters(self):
        expression, jaw = self.batch()
        betas = torch.randn(4, 100) * 0.5
        short = self.model(betas=betas, expression=expression[:, :50], jaw_pose=jaw)
        padded = self.model(betas=torch.cat([betas, torch.zeros(4, 200)], 1),
                            expression=torch.cat([expression[:, :50], torch.zeros(4, 50)], 1), jaw_pose=jaw)
        torch.testing.assert_close(short.vertices, padded.vertices)

    def test_flame_pytorch_and_deca_conventions(self):
        expression, jaw = self.batch()
        global_orient = torch.randn(4, 3) * 0.1
        eyes = torch.randn(4, 6) * 0.05
        reference = self.model(betas=torch.zeros(4, 100), expression=expression[:, :50], global_orient=global_orient,
                               jaw_pose=jaw, leye_pose=eyes[:, :3], reye_pose=eyes[:, 3:])
        pose_params = torch.cat([global_orient, jaw], dim=1)                    # FLAME_PyTorch: jaw at [3:6]
        flame_pytorch = self.model.from_flame_pytorch(torch.zeros(4, 100), expression[:, :50], pose_params,
                                                      eye_pose=eyes)
        torch.testing.assert_close(flame_pytorch.vertices, reference.vertices)
        deca = self.model.from_deca({"shape": torch.zeros(4, 100), "exp": expression[:, :50], "pose": pose_params})
        no_eyes = self.model(expression=expression[:, :50], global_orient=global_orient, jaw_pose=jaw)
        torch.testing.assert_close(deca.vertices, no_eyes.vertices)

    def test_muscle_driven_round_trip(self):
        expression, _ = self.batch()
        q_open = torch.tensor([0.05, 0.1, 0.3, 0.0], dtype=torch.float64)
        q_lateral = torch.tensor([0.0, 0.05, -0.1, 0.0], dtype=torch.float64)
        jaw = self.model.jaw_map.physical_to_theta(q_open, q_lateral).float()          # on the jaw manifold
        out = self.model(expression=expression, jaw_pose=jaw)
        again = self.model.from_activations(out.activations, jaw_lateral=out.jaw_lateral)
        torch.testing.assert_close(again.jaw_open, out.jaw_open, atol=1e-6, rtol=0)
        self.assertLess(float((again.vertices - out.vertices).abs().max()), 2e-5)

    def test_out_of_range_jaw(self):
        jaw = torch.tensor([[math.radians(35.0), 0.0, 0.0]])
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            out = self.model(expression=torch.from_numpy(self.expressions["expression"][self.index[:1]]), jaw_pose=jaw)
        self.assertFalse(bool(out.jaw_in_domain[0]))
        flame = self.model.flame_forward(jaw_pose=jaw)
        lower_lip = 3533                                                     # chin keeps FLAME's jaw rotation
        self.assertLess(float((out.vertices[0, lower_lip] - flame[0, lower_lip]).norm()), 0.01)

    def test_gradients(self):
        expression, jaw = self.batch()
        expression = expression.clone().requires_grad_(True)
        self.model(expression=expression, jaw_pose=jaw).vertices.square().sum().backward()
        self.assertTrue(bool(torch.isfinite(expression.grad).all()))
        a = torch.full((2, 50), 0.2, requires_grad=True)
        self.model.from_activations(a).vertices.square().sum().backward()
        self.assertGreater(float(a.grad.abs().sum()), 0.0)

    def test_exact_mode_close_to_surrogate(self):
        expression, jaw = self.batch()
        surrogate = self.model(expression=expression[:2], jaw_pose=jaw[:2]).activations
        exact = self.model(expression=expression[:2], jaw_pose=jaw[:2], activation_mode="exact").activations
        self.assertLess(float((surrogate - exact).abs().mean()), 0.03)

    def test_to_flame(self):
        expression, jaw = self.batch()
        out = self.model(expression=expression, jaw_pose=jaw)
        fitted = self.model.to_flame(out)
        self.assertEqual(tuple(fitted["expression"].shape), (4, 100))
        self.assertLess(float(fitted["rms"].max()), 1e-3)                      # below a millimetre

    def test_bilateral_symmetry(self):
        names = self.model.muscle_names
        left, right = names.index("Zygomaticus_Major_L"), names.index("Zygomaticus_Major_R")
        a = torch.zeros(2, 50)
        a[0, left] = 0.6
        a[1, right] = 0.6
        v = self.model.from_activations(a).vertices - self.model.flame_forward(batch_size=2)
        mirrored = v[1] * torch.tensor([-1.0, 1.0, 1.0])
        flame_x = self.model.flame_forward().squeeze(0)
        partner = torch.cdist(flame_x * torch.tensor([-1.0, 1.0, 1.0]), flame_x).argmin(1)
        error = (v[0] - mirrored[partner]).norm(dim=-1).max() / v[0].norm(dim=-1).max()
        self.assertLess(float(error), 0.15)


if __name__ == "__main__":
    unittest.main()
