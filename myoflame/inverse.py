"""Exact inverse physics: the box-constrained program the training labels come from.

For one sample the activations solve

    min_a || R(q) a - d ||^2 + lambda || diag(s) a ||^2    s.t.  0 <= a <= 1,  a_jaw = 0

where d is FLAME's expression displacement at the observed vertices (expression
on minus expression off, same identity and jaw pose), R(q) the response basis at
the physical jaw coordinates, s the column norms and lambda = 1e-4. It needs no
training and accepts any FLAME 2020 expression, at the cost of one bounded least
squares solve per sample on the CPU (SciPy BVLS); it is not differentiable.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import lsq_linear

REGULARISATION = 1e-4


def solve_box_qp(R: np.ndarray, d: np.ndarray, free: np.ndarray, reg: float = REGULARISATION) -> np.ndarray:
    """Column-normalised ridge BVLS on the free columns; other columns stay zero."""
    R = np.asarray(R, dtype=np.float64)
    norms = np.linalg.norm(R, axis=0)
    scales = np.where(norms > 1e-10, norms, 1.0)[free]
    A = np.vstack([R[:, free] / scales, np.sqrt(reg) * np.eye(len(free))])
    b = np.concatenate([np.asarray(d, dtype=np.float64), np.zeros(len(free))])
    result = lsq_linear(A, b, bounds=(np.zeros(len(free)), scales), method="bvls", tol=1e-10,
                        max_iter=max(200, 5 * len(free)))
    if not result.success:
        raise RuntimeError(f"inverse physics solve failed: {result.message}")
    a = np.zeros(R.shape[1])
    a[free] = np.clip(result.x / scales, 0.0, 1.0)
    return a


@torch.no_grad()
def exact_activations(model, betas, expression, jaw_pose, q_open, q_lateral) -> torch.Tensor:
    """Inverse-physics activations (B, 50) for a batch; the jaw columns are left at zero."""
    B = betas.shape[0]
    jaw_only = torch.zeros(B, 15, dtype=betas.dtype, device=betas.device)
    jaw_only[:, 6:9] = jaw_pose
    rest = model.flame(betas, None, jaw_only)
    posed = model.flame(betas, expression, jaw_only)
    d = (posed - rest)[:, model.observed].reshape(B, -1).double().cpu().numpy()
    free = np.array(model.vocabulary.soft_tissue)
    open_bins = model.jaw_open_bins.double().cpu().numpy()
    lateral_bins = model.jaw_lateral_bins.double().cpu().numpy()
    n_lateral = len(lateral_bins)
    R_cells = model.R_cells.cpu().numpy()
    out = np.zeros((B, model.n_muscles), dtype=np.float32)
    for b in range(B):
        if not np.any(expression[b].cpu().numpy()):
            continue
        qo = float(np.clip(q_open[b].item(), open_bins[0], open_bins[-1]))
        ql = float(np.clip(q_lateral[b].item(), lateral_bins[0], lateral_bins[-1]))
        i = min(max(int(np.searchsorted(open_bins, qo, side="right") - 1), 0), len(open_bins) - 2)
        j = min(max(int(np.searchsorted(lateral_bins, ql, side="right") - 1), 0), n_lateral - 2)
        u = (qo - open_bins[i]) / (open_bins[i + 1] - open_bins[i])
        v = (ql - lateral_bins[j]) / (lateral_bins[j + 1] - lateral_bins[j])
        R = None
        for (ci, cj), w in (((i, j), (1 - u) * (1 - v)), ((i, j + 1), (1 - u) * v),
                            ((i + 1, j), u * (1 - v)), ((i + 1, j + 1), u * v)):
            if w == 0.0:
                continue
            term = R_cells[ci * n_lateral + cj].astype(np.float64) * w
            R = term if R is None else R + term
        out[b] = solve_box_qp(R, d[b], free)
    return torch.from_numpy(out).to(betas.device)
