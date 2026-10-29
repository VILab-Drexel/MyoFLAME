"""MyoFLAME: a muscle-driven companion to FLAME with the same inputs and topology.

Given FLAME parameters, MyoFLAME estimates the activations of 50 named facial and
jaw muscles and returns the skin those muscles produce, in FLAME topology. The
identity, head, neck, eye and jaw joints are FLAME's own; the expression
blendshapes are replaced by the muscle response:

    v = FLAME(beta, psi = 0, pose)  +  T(pose) R(q_open, q_lateral) a

* ``a`` (B, 50): activations in [0, 1] from the surrogate (or from the exact
  inverse-physics program). The six jaw columns follow the jaw muscle law.
* ``R(q)``: the muscle response basis, metres per unit activation at 3,931
  FLAME vertices, on a 5 x 5 grid of physical jaw coordinates and bilinearly
  interpolated in between. The jaw columns move the skin through the mandible
  (FLAME's jaw joint), not through R.
* ``T(pose)``: carries the responses from the jaw-only pose into the full
  head, neck and eye pose (the linear part of FLAME's skinning).
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .flame import N_EXPRESSION, N_SHAPE, FLAMELayer, load_flame
from .jaw import JawCalibrationMap, JawMuscleLaw
from .surrogate import ActivationSurrogate, bilinear_weights
from .vocabulary import Vocabulary

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_FLAME_PATH = Path("model/FLAME2020/generic_model.pkl")
POSE_PARTS = ("global_orient", "neck_pose", "jaw_pose", "leye_pose", "reye_pose")


@dataclass
class MyoFLAMEOutput:
    """Batched result; every field has the batch size B as its first dimension."""

    vertices: torch.Tensor          # (B, 5023, 3) muscle-driven skin, FLAME topology
    activations: torch.Tensor       # (B, 50) muscle activations in [0, 1], order of model.muscle_names
    jaw_open: torch.Tensor          # (B,) physical jaw opening, radians
    jaw_lateral: torch.Tensor       # (B,) physical lateral excursion, radians
    jaw_in_domain: torch.Tensor     # (B,) bool, False if the jaw pose lies outside the calibrated range
    full_pose: torch.Tensor         # (B, 15) global, neck, jaw, left eye, right eye (axis-angle)
    betas: torch.Tensor             # (B, 300)
    transl: torch.Tensor            # (B, 3)
    flame_vertices: torch.Tensor | None = None   # (B, 5023, 3) plain FLAME for the same input, if requested

    def numpy(self) -> dict:
        """The fields as NumPy arrays."""
        return {f.name: getattr(self, f.name).detach().cpu().numpy()
                for f in fields(self) if getattr(self, f.name) is not None}


def _inverse3x3(A: torch.Tensor) -> torch.Tensor:
    """Adjugate inverse of (..., 3, 3) matrices in float64."""
    a = A.to(torch.float64)
    m = [[a[..., r, c] for c in range(3)] for r in range(3)]
    c00 = m[1][1] * m[2][2] - m[1][2] * m[2][1]
    c01 = m[1][2] * m[2][0] - m[1][0] * m[2][2]
    c02 = m[1][0] * m[2][1] - m[1][1] * m[2][0]
    det = m[0][0] * c00 + m[0][1] * c01 + m[0][2] * c02
    rows = [
        [c00, m[0][2] * m[2][1] - m[0][1] * m[2][2], m[0][1] * m[1][2] - m[0][2] * m[1][1]],
        [c01, m[0][0] * m[2][2] - m[0][2] * m[2][0], m[0][2] * m[1][0] - m[0][0] * m[1][2]],
        [c02, m[0][1] * m[2][0] - m[0][0] * m[2][1], m[0][0] * m[1][1] - m[0][1] * m[1][0]],
    ]
    inv = torch.stack([torch.stack(r, dim=-1) for r in rows], dim=-2)
    return inv / det[..., None, None]


class MyoFLAME(nn.Module):
    """FLAME parameters in, muscle activations and muscle-driven FLAME-topology skin out.

    Parameters
    ----------
    flame_model_path : path to FLAME 2020 ``generic_model.pkl`` (not distributed with
        MyoFLAME; see model/FLAME2020/README.md).
    data_dir : directory with the MyoFLAME model files (default: the package data).
    check_flame_version : refuse models whose expression basis is not FLAME 2020's.
    """

    def __init__(self, flame_model_path=DEFAULT_FLAME_PATH, data_dir=DATA_DIR,
                 check_flame_version: bool = True):
        super().__init__()
        data_dir = Path(data_dir)
        self.flame = FLAMELayer(load_flame(flame_model_path, check_version=check_flame_version))
        self.info = json.loads((data_dir / "model_info.json").read_text())

        grid = np.load(data_dir / "muscle_responses.npz", allow_pickle=False)
        self.vocabulary = Vocabulary(grid["muscle_names"], grid["muscle_types"])
        R = torch.from_numpy(grid["R_grid"].astype(np.float32))           # (n_o, n_l, 3*n_obs, n_muscles)
        n_open, n_lateral, rows, n_muscles = R.shape
        self.register_buffer("R_cells", R.reshape(n_open * n_lateral, rows, n_muscles).contiguous())
        self.grid_shape = (n_open, n_lateral)
        self.register_buffer("jaw_open_bins", torch.from_numpy(grid["jaw_open_bins"].astype(np.float32)))
        self.register_buffer("jaw_lateral_bins", torch.from_numpy(grid["jaw_lateral_bins"].astype(np.float32)))
        self.register_buffer("observed", torch.from_numpy(grid["flame_obs_idx"].astype(np.int64)))
        skin_mask = torch.ones(n_muscles)
        skin_mask[self.vocabulary.jaw] = 0.0                                # jaw columns move the mandible
        self.register_buffer("skin_mask", skin_mask)

        self.jaw_map = JawCalibrationMap(dict(np.load(data_dir / "jaw_calibration.npz", allow_pickle=False)))
        self.jaw_law = JawMuscleLaw(self.vocabulary.names, self.jaw_map.q_open_max)
        self.surrogate = ActivationSurrogate.from_checkpoint(data_dir / "surrogate.pt")
        self._warned_domain = False

    # ------------------------------------------------------------------ properties
    @property
    def muscle_names(self) -> list[str]:
        return self.vocabulary.names

    @property
    def muscle_types(self) -> list[str]:
        return self.vocabulary.types

    @property
    def n_muscles(self) -> int:
        return len(self.vocabulary)

    @property
    def faces(self) -> torch.Tensor:
        return self.flame.faces_tensor

    # ------------------------------------------------------------------ input handling
    def _batch(self, tensors, batch_size=None) -> tuple[int, torch.device]:
        device = self.R_cells.device
        sizes = {int(t.shape[0]) for t in tensors if t is not None}
        if batch_size is not None:
            sizes.add(int(batch_size))
        if len(sizes) > 1:
            raise ValueError(f"inconsistent batch sizes {sorted(sizes)}")
        return (sizes.pop() if sizes else 1), device

    def _as_tensor(self, x, width=None, name=""):
        if x is None:
            return None
        t = torch.as_tensor(x, dtype=torch.float32, device=self.R_cells.device)
        if t.ndim == 1:
            t = t[None]
        if t.ndim != 2:
            raise ValueError(f"{name} must have shape (B, n), got {tuple(t.shape)}")
        if width is not None and t.shape[1] != width:
            raise ValueError(f"{name} must have {width} values per row, got {t.shape[1]}")
        return t

    @staticmethod
    def _pad(t, width, B, device, name):
        if t is None:
            return torch.zeros(B, width, device=device)
        if t.shape[1] > width:
            raise ValueError(f"{name} has {t.shape[1]} coefficients, FLAME 2020 has {width}")
        if t.shape[1] < width:
            t = torch.cat([t, torch.zeros(t.shape[0], width - t.shape[1], device=t.device, dtype=t.dtype)], dim=1)
        return t

    def _parse(self, betas, expression, global_orient, neck_pose, jaw_pose, leye_pose, reye_pose, transl,
               batch_size=None):
        betas = self._as_tensor(betas, name="betas")
        expression = self._as_tensor(expression, name="expression")
        parts = [self._as_tensor(p, 3, name) for p, name in zip(
            (global_orient, neck_pose, jaw_pose, leye_pose, reye_pose), POSE_PARTS, strict=True)]
        transl = self._as_tensor(transl, 3, "transl")
        B, device = self._batch([betas, expression, transl, *parts], batch_size)
        betas = self._pad(betas, N_SHAPE, B, device, "betas")
        expression = self._pad(expression, N_EXPRESSION, B, device, "expression")
        full_pose = torch.cat([p if p is not None else torch.zeros(B, 3, device=device) for p in parts], dim=1)
        transl = transl if transl is not None else torch.zeros(B, 3, device=device)
        return betas, expression, full_pose, transl

    # ------------------------------------------------------------------ pieces
    def project_jaw(self, jaw_pose: torch.Tensor):
        """Physical jaw coordinates of FLAME jaw poses (B,3): q_open, q_lateral (float64) and in-domain flags."""
        q, in_domain, _ = self.jaw_map.project(jaw_pose.detach())
        if not bool(in_domain.all()) and not self._warned_domain:
            warnings.warn(
                "some jaw poses lie outside the calibrated jaw range (opening 0 to 20 degrees, lateral -12 to "
                "12 degrees); the skin keeps the given jaw rotation, the activations use the nearest calibrated "
                "jaw position (see jaw_in_domain)", stacklevel=3)
            self._warned_domain = True
        return q[:, 0], q[:, 1], in_domain

    def muscle_displacement(self, betas, activations, q_open, q_lateral, full_pose, chunk: int = 256, J=None):
        """Skin displacement (B, n_obs, 3) of the soft-tissue activations at the posed observed vertices."""
        B = activations.shape[0]
        i, j, u, v = bilinear_weights(self.jaw_open_bins.double(), self.jaw_lateral_bins.double(),
                                      q_open.double(), q_lateral.double())
        n_lateral = self.grid_shape[1]
        corners = torch.stack([i * n_lateral + j, i * n_lateral + j + 1,
                               (i + 1) * n_lateral + j, (i + 1) * n_lateral + j + 1], dim=1)
        weights = torch.stack([(1 - u) * (1 - v), (1 - u) * v, u * (1 - v), u * v], dim=1).float()
        a = activations * self.skin_mask
        out = []
        for start in range(0, B, chunk):
            sl = slice(start, start + chunk)
            used, local = torch.unique(corners[sl], return_inverse=True)
            Y = torch.einsum("cnm,bm->bcn", self.R_cells[used], a[sl])          # (b, cells used, 3*n_obs)
            picked = torch.gather(Y, 1, local[:, :, None].expand(-1, -1, Y.shape[2]))
            out.append((picked * weights[sl, :, None]).sum(1))
        disp = torch.cat(out).reshape(B, -1, 3)
        # carry the responses from the jaw-only pose into the full pose
        jaw_reference = self.jaw_map.physical_to_theta(q_open, q_lateral).float()
        jaw_only = torch.zeros_like(full_pose)
        jaw_only[:, 6:9] = jaw_reference
        if J is None:
            _, J = self.flame.shaped(betas)
        A_jaw = self.flame.blend_rotations(J, jaw_only, self.observed)
        A_full = self.flame.blend_rotations(J, full_pose, self.observed)
        transport = (A_full.double() @ _inverse3x3(A_jaw)).float()
        return (transport @ disp[..., None])[..., 0]

    def _skin(self, betas, full_pose, transl, activations, q_open, q_lateral):
        shaped = self.flame.shaped(betas)
        rest = self.flame(betas, None, full_pose, transl, shaped=shaped)
        disp = self.muscle_displacement(betas, activations, q_open, q_lateral, full_pose, J=shaped[1])
        return rest.index_add(1, self.observed, disp)

    # ------------------------------------------------------------------ public API
    def forward(self, betas=None, expression=None, global_orient=None, neck_pose=None, jaw_pose=None,
                leye_pose=None, reye_pose=None, transl=None, *, activation_mode: str = "surrogate",
                return_flame: bool = False, batch_size=None) -> MyoFLAMEOutput:
        """FLAME parameters to activations and muscle-driven skin.

        Arguments use the names of ``smplx.FLAME``; each is (B, n) and optional (zeros).
        ``betas`` and ``expression`` may be truncated (e.g. 100 shape / 50 expression
        coefficients); they are zero padded to FLAME 2020's 300 / 100.
        ``activation_mode``: "surrogate" (batched, differentiable) or "exact" (the
        inverse-physics program per sample, NumPy/SciPy, not differentiable).
        """
        betas, expression, full_pose, transl = self._parse(
            betas, expression, global_orient, neck_pose, jaw_pose, leye_pose, reye_pose, transl, batch_size)
        q_open, q_lateral, in_domain = self.project_jaw(full_pose[:, 6:9])
        if activation_mode == "surrogate":
            a = self.surrogate(betas, expression, q_open.float(), q_lateral.float())
            neutral = ~(expression != 0).any(dim=1)
            a = torch.where(neutral[:, None], torch.zeros_like(a), a)
        elif activation_mode == "exact":
            from .inverse import exact_activations
            a = exact_activations(self, betas, expression, full_pose[:, 6:9], q_open, q_lateral)
        else:
            raise ValueError(f"activation_mode must be 'surrogate' or 'exact', got {activation_mode!r}")
        a = self.jaw_law.fill(a, q_open)
        vertices = self._skin(betas, full_pose, transl, a, q_open, q_lateral)
        flame_vertices = self.flame(betas, expression, full_pose, transl) if return_flame else None
        return MyoFLAMEOutput(vertices, a, q_open.float(), q_lateral.float(), in_domain, full_pose, betas,
                              transl, flame_vertices)


    def from_activations(self, activations, betas=None, jaw_lateral=None, global_orient=None, neck_pose=None,
                         leye_pose=None, reye_pose=None, transl=None) -> MyoFLAMEOutput:
        """Muscle-driven skin: the activations alone set the face.

        The jaw muscles set the jaw opening through the jaw muscle law; ``jaw_lateral``
        (B,) radians is the lateral excursion (default 0). Activations are clamped to
        [0, 1]. Differentiable with respect to activations and betas.
        """
        a = self._as_tensor(activations, self.n_muscles, "activations").clamp(0.0, 1.0)
        betas, _, full_pose, transl = self._parse(betas, None, global_orient, neck_pose, None, leye_pose,
                                                  reye_pose, transl, a.shape[0])
        B = a.shape[0]
        q_lateral = (torch.zeros(B, device=a.device, dtype=torch.float64) if jaw_lateral is None
                     else torch.as_tensor(jaw_lateral, device=a.device, dtype=torch.float64).reshape(B))
        q_open = self.jaw_law.q_open(a.double())
        full_pose = full_pose.clone()
        full_pose[:, 6:9] = self.jaw_map.physical_to_theta(q_open, q_lateral).float()
        vertices = self._skin(betas, full_pose, transl, a, q_open, q_lateral)
        in_domain = torch.ones(B, dtype=torch.bool, device=a.device)
        return MyoFLAMEOutput(vertices, a, q_open.float(), q_lateral.float(), in_domain, full_pose, betas, transl)

    def flame_forward(self, betas=None, expression=None, global_orient=None, neck_pose=None, jaw_pose=None,
                      leye_pose=None, reye_pose=None, transl=None, batch_size=None) -> torch.Tensor:
        """Plain FLAME 2020 vertices (B, 5023, 3) for the same arguments."""
        betas, expression, full_pose, transl = self._parse(
            betas, expression, global_orient, neck_pose, jaw_pose, leye_pose, reye_pose, transl, batch_size)
        return self.flame(betas, expression, full_pose, transl)

    def to_flame(self, output: MyoFLAMEOutput, iterations: int = 3, chunk: int = 32) -> dict:
        """FLAME parameters whose skin is closest to MyoFLAME's (least squares over all vertices).

        Returns ``expression`` (B, 100), ``jaw_pose`` (B, 3) and the per-sample RMS
        vertex distance ``rms`` (B,) in metres; identity, pose and translation are
        the output's own. Use this to feed muscle-driven results to tools that only
        understand FLAME parameters.
        """
        target = output.vertices.detach()
        betas, pose, transl = output.betas.detach(), output.full_pose.detach(), output.transl.detach()
        expressions = []
        for start in range(0, target.shape[0], chunk):
            sl = slice(start, start + chunk)
            psi = torch.zeros(target[sl].shape[0], N_EXPRESSION, device=target.device)

            def skin(p, b, q, t):
                return self.flame(b[None], p[None], q[None], t[None])[0].reshape(-1)

            for _ in range(iterations):
                jac = torch.vmap(torch.func.jacfwd(skin), in_dims=(0, 0, 0, 0))(psi, betas[sl], pose[sl], transl[sl])
                residual = (target[sl].reshape(psi.shape[0], -1)
                            - torch.vmap(skin)(psi, betas[sl], pose[sl], transl[sl]))
                normal = jac.transpose(1, 2) @ jac
                rhs = jac.transpose(1, 2) @ residual[..., None]
                psi = psi + torch.linalg.solve(normal.double(), rhs.double())[..., 0].float()
            expressions.append(psi)
        expression = torch.cat(expressions)
        fitted = self.flame(betas, expression, pose, transl)
        rms = (fitted - target).norm(dim=-1).pow(2).mean(-1).sqrt()
        return {"expression": expression, "jaw_pose": pose[:, 6:9].clone(), "rms": rms}

    # ------------------------------------------------------------------ adapters
    def from_flame_pytorch(self, shape_params=None, expression_params=None, pose_params=None, neck_pose=None,
                           eye_pose=None, transl=None, **kwargs) -> MyoFLAMEOutput:
        """Arguments of FLAME_PyTorch: ``pose_params`` (B,6) = global (3) then jaw (3); ``eye_pose`` (B,6)."""
        pose = self._as_tensor(pose_params, 6, "pose_params")
        eyes = self._as_tensor(eye_pose, 6, "eye_pose")
        return self(betas=shape_params, expression=expression_params,
                    global_orient=None if pose is None else pose[:, :3],
                    jaw_pose=None if pose is None else pose[:, 3:6], neck_pose=neck_pose,
                    leye_pose=None if eyes is None else eyes[:, :3], reye_pose=None if eyes is None else eyes[:, 3:],
                    transl=transl, **kwargs)

    def from_deca(self, codedict: dict, **kwargs) -> MyoFLAMEOutput:
        """A DECA / EMOCA code dictionary: ``shape`` (B,100), ``exp`` (B,50), ``pose`` (B,6) = global then jaw."""
        return self.from_flame_pytorch(shape_params=codedict["shape"], expression_params=codedict["exp"],
                                       pose_params=codedict["pose"], **kwargs)
