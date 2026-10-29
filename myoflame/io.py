"""OBJ files and the bundled D3DFACS expression descriptors."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .model import DATA_DIR


def save_obj(path, vertices, faces) -> None:
    """Write a triangle mesh (vertices (V,3), faces (F,3) zero-based) as OBJ."""
    vertices = np.asarray(vertices.detach().cpu() if hasattr(vertices, "detach") else vertices)
    faces = np.asarray(faces.detach().cpu() if hasattr(faces, "detach") else faces)
    with open(path, "w") as handle:
        handle.writelines(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in vertices)
        handle.writelines(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in faces)


def load_expressions(data_dir=DATA_DIR) -> dict:
    """The 218 FACS-coded D3DFACS expressions as FLAME 2020 parameters.

    Returns ``label`` (AU code, e.g. "12" or "1+2"), ``name``, ``expression`` (218,100),
    ``jaw_pose`` (218,3) and ``n_subjects`` (how many D3DFACS subjects the descriptor averages).
    """
    data = np.load(Path(data_dir) / "expressions.npz", allow_pickle=False)
    return {key: data[key] for key in data.files}


def load_action_units(data_dir=DATA_DIR) -> dict:
    """FACS action unit -> name and the muscle columns that perform it."""
    return json.loads((Path(data_dir) / "action_units.json").read_text())
