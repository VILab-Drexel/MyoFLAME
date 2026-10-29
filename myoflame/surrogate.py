"""The activation surrogate g_phi: FLAME expression and jaw to muscle activations.

    a = clip(W(jaw) psi + g(|psi|) alpha * tanh(MLP(x)), 0, 1)

``W(jaw)`` is a linear prior (one 50 x 100 matrix per cell of the 5 x 5 jaw grid,
bilinearly interpolated at the physical jaw coordinates), ``MLP`` is a residual
network on x = (beta / beta_std, psi / 0.7, q_open / 40 deg, q_lateral / 12 deg, 0)
and g(r) = 1 - exp(-r^2 / sigma^2) fades the residual out towards the neutral face.
The six jaw columns are masked: their values come from the jaw muscle law.

It was trained to reproduce the inverse-physics activations of 218 FACS-coded
D3DFACS expressions on 1,211 LYHM identities.
"""
from __future__ import annotations

import math

import torch
from torch import nn

Q_OPEN_SCALE = 40.0 * 3.14159265358979 / 180.0
Q_LATERAL_SCALE = 12.0 * 3.14159265358979 / 180.0
Q_PROTRUSION_SCALE = 0.014
PSI_SCALE = 0.7


def bilinear_weights(open_bins: torch.Tensor, lateral_bins: torch.Tensor,
                     q_open: torch.Tensor, q_lateral: torch.Tensor):
    """Lower cell indices (i, j) and fractions (u, v) of a bilinear grid lookup, clamped to the grid."""
    qo = torch.clamp(q_open, open_bins[0], open_bins[-1])
    ql = torch.clamp(q_lateral, lateral_bins[0], lateral_bins[-1])
    i = (torch.searchsorted(open_bins, qo.contiguous(), right=True) - 1).clamp(0, open_bins.shape[0] - 2)
    j = (torch.searchsorted(lateral_bins, ql.contiguous(), right=True) - 1).clamp(0, lateral_bins.shape[0] - 2)
    u = (qo - open_bins[i]) / (open_bins[i + 1] - open_bins[i]).clamp(min=1e-12)
    v = (ql - lateral_bins[j]) / (lateral_bins[j + 1] - lateral_bins[j]).clamp(min=1e-12)
    return i, j, u, v


class ActivationSurrogate(nn.Module):
    def __init__(self, W_ph_grid: torch.Tensor, jaw_open_bins: torch.Tensor, jaw_lateral_bins: torch.Tensor,
                 input_dim: int = 403, hidden=(256, 256, 128)):
        super().__init__()
        n_muscles = W_ph_grid.shape[2]
        self.register_buffer("W_ph_grid", W_ph_grid)
        self.register_buffer("jaw_open_bins", jaw_open_bins)
        self.register_buffer("jaw_lateral_bins", jaw_lateral_bins)
        self.alpha = nn.Parameter(torch.zeros(n_muscles))
        self.register_buffer("mask_c", torch.ones(n_muscles))
        self.register_buffer("neutral_gate_sigma", torch.tensor(1.0))
        layers, width = [], input_dim
        for h in hidden:
            layers += [nn.Linear(width, h), nn.LayerNorm(h), nn.GELU()]
            width = h
        layers.append(nn.Linear(width, n_muscles))
        self.mlp = nn.Sequential(*layers)
        self.register_buffer("beta_std", torch.ones(300))

    def inputs(self, betas: torch.Tensor, psi: torch.Tensor, q_open: torch.Tensor, q_lateral: torch.Tensor):
        q = torch.stack([q_open / Q_OPEN_SCALE, q_lateral / Q_LATERAL_SCALE,
                         torch.zeros_like(q_open)], dim=-1)
        return torch.cat([betas / self.beta_std.clamp(min=1e-3), psi / PSI_SCALE, q], dim=-1)

    def forward(self, betas: torch.Tensor, psi: torch.Tensor, q_open: torch.Tensor,
                q_lateral: torch.Tensor) -> torch.Tensor:
        """Activations (B, 50) from betas (B,300), psi (B,100) and physical jaw coordinates (B,) in radians."""
        i, j, u, v = bilinear_weights(self.jaw_open_bins, self.jaw_lateral_bins, q_open, q_lateral)
        u, v = u[:, None, None], v[:, None, None]
        W = ((1 - u) * (1 - v) * self.W_ph_grid[i, j] + (1 - u) * v * self.W_ph_grid[i, j + 1]
             + u * (1 - v) * self.W_ph_grid[i + 1, j] + u * v * self.W_ph_grid[i + 1, j + 1])
        linear = torch.einsum("bmk,bk->bm", W, psi)
        residual = torch.tanh(self.mlp(self.inputs(betas, psi, q_open, q_lateral))) * self.alpha
        sigma = float(self.neutral_gate_sigma)
        if sigma > 0:
            gate = -torch.expm1(-(psi * psi).sum(-1, keepdim=True) / (sigma * sigma))
            residual = residual * gate
        return (linear + residual).clamp(0.0, 1.0) * self.mask_c

    @classmethod
    def from_checkpoint(cls, path, map_location="cpu") -> ActivationSurrogate:
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
        state = dict(checkpoint["state_dict"])
        model = cls(state["W_ph_grid"], state["jaw_open_bins"], state["jaw_lateral_bins"])
        state["beta_std"] = torch.as_tensor(checkpoint["beta_std"], dtype=torch.float32)
        model.load_state_dict(state)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        if not math.isfinite(float(model.neutral_gate_sigma)):
            raise ValueError("invalid neutral gate width in the surrogate checkpoint")
        return model
