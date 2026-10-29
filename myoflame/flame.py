"""FLAME 2020: pickle loader (no chumpy) and a batched PyTorch forward pass.

The FLAME 2020 model file (``generic_model.pkl``) is not part of this repository.
Register at https://flame.is.tue.mpg.de, agree to the FLAME model licence and
copy the file into ``model/FLAME2020/`` (or pass its path explicitly).

The pickle stores ``shapedirs`` as a ``chumpy`` object. A restricted unpickler
maps that one class onto a small stand-in, so chumpy is not needed, and refuses
every global outside NumPy, SciPy's sparse matrices and the builtin containers.
"""
from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np
import torch

N_VERTICES = 5023
N_FACES = 9976
N_SHAPE = 300
N_EXPRESSION = 100
N_JOINTS = 5
JOINT_NAMES = ("global", "neck", "jaw", "left_eye", "right_eye")

# Three scalars of the FLAME 2020 expression basis (Frobenius norms of expression
# components 1, 2 and 100). The activation surrogate was trained in FLAME 2020's
# expression space; FLAME 2023 has the same array shapes but a different
# expression basis, so a shape check cannot tell the two apart.
FLAME2020_EXPRESSION_NORMS = (0.0910598, 0.0501290, 0.00148359)


class _ChumpyArray:
    """Stand-in for ``chumpy.ch.Ch``: keeps the pickled state; ``.x`` is the array."""

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __array__(self, dtype=None, copy=None):
        x = getattr(self, "x", None)
        return np.array(x, dtype=dtype) if x is not None else np.array([])


_TRUSTED_BUILTINS = {"set", "frozenset", "dict", "list", "tuple", "object"}


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.split(".")[0] == "chumpy":
            if name == "Ch":
                return _ChumpyArray
            raise pickle.UnpicklingError(f"unexpected chumpy symbol {module}.{name}")
        if module in ("builtins", "__builtin__"):
            if name in _TRUSTED_BUILTINS:
                return getattr(__import__("builtins"), name)
            raise pickle.UnpicklingError(f"refusing builtin {name} while loading FLAME")
        if module.startswith("scipy.sparse"):
            import scipy.sparse
            return getattr(scipy.sparse, name)
        if module.split(".")[0] == "numpy":
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refusing global {module}.{name} while loading FLAME")


def _load_pickle(path: Path):
    with path.open("rb") as handle, warnings.catch_warnings():
        # NumPy 2.4 deprecates the integer ``align`` the Python 2 pickles pass to dtype().
        warnings.filterwarnings("ignore", message=r"dtype\(\): align should be passed")
        return _RestrictedUnpickler(handle, encoding="latin1").load()


def _missing_flame_message(path: Path) -> str:
    return (
        f"FLAME model not found: {path}\n"
        "MyoFLAME needs the FLAME 2020 model, which cannot be redistributed. Register at "
        "https://flame.is.tue.mpg.de, agree to the model licence, download FLAME 2020 and "
        "copy generic_model.pkl into model/FLAME2020/ (see model/FLAME2020/README.md)."
    )


def load_flame(path, check_version: bool = True) -> dict:
    """Load FLAME 2020 into plain NumPy arrays.

    Returns a dict with ``v_template (5023,3)``, ``shapedirs (5023,3,400)``,
    ``posedirs (5023,3,36)``, ``J_regressor (5,5023)``, ``weights (5023,5)``,
    ``faces (9976,3)`` and ``parents (5,)``.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(_missing_flame_message(path))
    raw = _load_pickle(path)
    regressor = raw["J_regressor"]
    flame = {
        "v_template": np.asarray(raw["v_template"], dtype=np.float32),
        "shapedirs": np.asarray(np.array(raw["shapedirs"]), dtype=np.float32),
        "posedirs": np.asarray(raw["posedirs"], dtype=np.float32),
        "J_regressor": np.asarray(regressor.toarray() if hasattr(regressor, "toarray") else regressor,
                                  dtype=np.float32),
        "weights": np.asarray(raw["weights"], dtype=np.float32),
        "faces": np.asarray(raw["f"], dtype=np.int64),
        "parents": np.asarray(raw["kintree_table"]).astype(np.int32).astype(np.int64)[0],   # root stored as uint32 -1
    }
    expected = {
        "v_template": (N_VERTICES, 3),
        "shapedirs": (N_VERTICES, 3, N_SHAPE + N_EXPRESSION),
        "posedirs": (N_VERTICES, 3, 9 * (N_JOINTS - 1)),
        "J_regressor": (N_JOINTS, N_VERTICES),
        "weights": (N_VERTICES, N_JOINTS),
        "faces": (N_FACES, 3),
    }
    for key, shape in expected.items():
        if flame[key].shape != shape:
            raise ValueError(f"{path} is not a FLAME 2020 model: {key} has shape "
                             f"{flame[key].shape}, expected {shape}")
    if check_version:
        norms = [float(np.linalg.norm(flame["shapedirs"][:, :, N_SHAPE + k])) for k in (0, 1, 99)]
        if not np.allclose(norms, FLAME2020_EXPRESSION_NORMS, rtol=1e-3, atol=0.0):
            raise ValueError(
                f"{path} does not carry the FLAME 2020 expression basis (norms {norms}). "
                "MyoFLAME's activation surrogate is defined in FLAME 2020's expression space; "
                "FLAME 2023 and other versions are not supported. Pass check_version=False only "
                "if you know the expression basis is the FLAME 2020 one."
            )
    return flame


def load_flame_masks(path) -> dict:
    """Official FLAME vertex masks (``FLAME_masks.pkl``): region name -> vertex indices."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"FLAME masks not found: {path} (FLAME_masks.pkl of the FLAME 2020 release)")
    raw = _load_pickle(path)
    return {str(k): np.unique(np.asarray(v, dtype=np.int64)) for k, v in raw.items()}


# ---------------------------------------------------------------------------
# batched PyTorch forward

def axis_angle_to_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    """Rodrigues' formula, (..., 3) axis-angle -> (..., 3, 3), stable at zero angle."""
    angle2 = (rotvec * rotvec).sum(-1, keepdim=True)
    angle = torch.sqrt(angle2.clamp(min=1e-30))
    small = angle2 < 1e-12
    # sin(t)/t and (1-cos(t))/t^2 with their Taylor series near zero
    a = torch.where(small, 1.0 - angle2 / 6.0, torch.sin(angle) / angle)
    b = torch.where(small, 0.5 - angle2 / 24.0, (1.0 - torch.cos(angle)) / angle2.clamp(min=1e-30))
    x, y, z = rotvec.unbind(-1)
    zero = torch.zeros_like(x)
    K = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], dim=-1).reshape(rotvec.shape[:-1] + (3, 3))
    eye = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device).expand(K.shape)
    return eye + a[..., None] * K + b[..., None] * (K @ K)


def joint_transforms(J: torch.Tensor, rot: torch.Tensor, parents) -> torch.Tensor:
    """Rest-relative joint transforms (B, n_joints, 4, 4) from joints J (B,n,3) and rotations (B,n,3,3).

    Written without in-place updates so that it composes with torch.func transforms.
    """
    B, n = J.shape[:2]
    bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=J.dtype, device=J.device).expand(B, 1, 4)
    G = []
    for i in range(n):
        p = int(parents[i])
        offset = J[:, i] if p < 0 else J[:, i] - J[:, p]
        local = torch.cat([torch.cat([rot[:, i], offset[:, :, None]], dim=2), bottom], dim=1)
        G.append(local if p < 0 else G[p] @ local)
    G = torch.stack(G, dim=1)
    # subtract the rest pose: G' = G @ T(-J)
    translation = G[..., :3, 3] - (G[..., :3, :3] @ J[..., None])[..., 0]
    top = torch.cat([G[..., :3, :3], translation[..., None]], dim=-1)
    return torch.cat([top, G[..., 3:, :]], dim=-2)


class FLAMELayer(torch.nn.Module):
    """Batched FLAME 2020 forward (shape, expression, five joints, linear blend skinning)."""

    def __init__(self, flame: dict):
        super().__init__()
        self.register_buffer("v_template", torch.from_numpy(flame["v_template"]))
        self.register_buffer("shapedirs", torch.from_numpy(flame["shapedirs"]))
        self.register_buffer("posedirs", torch.from_numpy(flame["posedirs"]).reshape(-1, 36).T.contiguous())
        self.register_buffer("J_regressor", torch.from_numpy(flame["J_regressor"]))
        self.register_buffer("lbs_weights", torch.from_numpy(flame["weights"]))
        self.register_buffer("faces_tensor", torch.from_numpy(flame["faces"]))
        self.parents = [int(p) for p in flame["parents"]]

    def shaped(self, betas: torch.Tensor, expression: torch.Tensor | None = None):
        """Shaped rest template (B,5023,3) and its joints (B,5,3)."""
        if expression is None:
            expression = betas.new_zeros(betas.shape[0], self.shapedirs.shape[2] - betas.shape[1])
        coeffs = torch.cat([betas, expression], dim=-1)
        v = self.v_template + torch.einsum("vik,bk->bvi", self.shapedirs, coeffs)
        J = torch.einsum("jv,bvi->bji", self.J_regressor, v)
        return v, J

    def transforms(self, J: torch.Tensor, full_pose: torch.Tensor):
        rot = axis_angle_to_matrix(full_pose.reshape(-1, N_JOINTS, 3))
        return joint_transforms(J, rot, self.parents), rot

    def forward(self, betas, expression, full_pose, transl=None, shaped=None):
        """Vertices (B,5023,3). ``full_pose`` is (B,15) = global, neck, jaw, left eye, right eye.

        ``shaped``: the result of ``self.shaped(betas, expression)`` if already computed."""
        v_shaped, J = self.shaped(betas, expression) if shaped is None else shaped
        G, rot = self.transforms(J, full_pose)
        eye = torch.eye(3, dtype=rot.dtype, device=rot.device)
        pose_feature = (rot[:, 1:] - eye).reshape(rot.shape[0], -1)
        v_posed = v_shaped + (pose_feature @ self.posedirs).reshape(v_shaped.shape)
        T = torch.einsum("vj,njxy->nvxy", self.lbs_weights, G)
        v = (T[..., :3, :3] @ v_posed[..., None])[..., 0] + T[..., :3, 3]
        if transl is not None:
            v = v + transl[:, None, :]
        return v

    def blend_rotations(self, J: torch.Tensor, full_pose: torch.Tensor, vertex_index: torch.Tensor):
        """Blended 3x3 linear part of the skinning transform at the given vertices, (B,V,3,3)."""
        G, _ = self.transforms(J, full_pose)
        R = G[..., :3, :3].reshape(G.shape[0], G.shape[1], 9)
        return (self.lbs_weights[vertex_index] @ R).reshape(G.shape[0], -1, 3, 3)
