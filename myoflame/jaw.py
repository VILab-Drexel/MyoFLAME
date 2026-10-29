"""Physical jaw coordinates and the jaw muscle law.

FLAME rotates the mandible with a free three-axis jaw joint. MyoFLAME describes the
jaw with two physical coordinates, the opening ``q_open`` (0 to 20 degrees) and the
lateral excursion ``q_lateral`` (-12 to 12 degrees). A calibration fitted FLAME's
jaw rotation to a rigid mandible rotating about the temporomandibular joints on a
5 x 5 grid of these coordinates; three bicubic splines interpolate the FLAME jaw
axis-angle over the grid (:class:`JawCalibrationMap`). An arbitrary FLAME jaw pose
is mapped to physical coordinates by projecting it onto this two-dimensional
manifold (bounded least squares).

The six jaw muscles drive the opening through one invertible law
(:class:`JawMuscleLaw`)::

    q_open = Q_max * clip(mean(a_digastric) - k_m mean(a_masseter) - k_t mean(a_temporalis), 0, 1)

with k_m = 5.23 and k_t = 2.10, the closers' torque capacity relative to the
jaw-depressor group (F_max x moment arm: masseter 8.60 N m, temporalis 3.45 N m,
digastric 1.65 N m). The digastric columns stand for the whole depressor group.
The minimum-effort inverse is a_digastric = q_open / Q_max on both sides with the
closers at zero. Lateral excursion stays a pose input.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.interpolate import RectBivariateSpline

# torque capacities (force x moment arm)
TORQUE_MASSETER = 215.0 * 0.040
TORQUE_TEMPORALIS = 115.0 * 0.030
TORQUE_DIGASTRIC = 35.0 * 0.047
KAPPA_MASSETER = TORQUE_MASSETER / TORQUE_DIGASTRIC       # 5.228
KAPPA_TEMPORALIS = TORQUE_TEMPORALIS / TORQUE_DIGASTRIC   # 2.097

OPENERS = ("Digastric_L", "Digastric_R")
MASSETER = ("Masseter_L", "Masseter_R")
TEMPORALIS = ("Temporalis_L", "Temporalis_R")

# a projection that has to move the jaw further than this outside the calibrated
# domain is reported as out of domain
OUT_OF_DOMAIN_TOLERANCE_RAD = float(np.radians(0.5))


def _bspline_basis(x: torch.Tensor, t: torch.Tensor, k: int):
    """All B-spline basis functions of degree k and their first derivatives at x.

    ``x`` (B,) inside [t[k], t[-k-1]]; returns (B, len(t)-k-1) values and derivatives.
    """
    n_knots = t.shape[0]
    span = torch.searchsorted(t, x.contiguous(), right=True) - 1
    span = span.clamp(k, n_knots - k - 2)
    basis = torch.nn.functional.one_hot(span, n_knots - 1).to(x.dtype)
    lower = None
    for d in range(1, k + 1):
        if d == k:
            lower = basis
        n_funcs = n_knots - 1 - d
        ti = t[:n_funcs]
        ti_d = t[d: d + n_funcs]
        ti1 = t[1: 1 + n_funcs]
        ti_d1 = t[d + 1: d + 1 + n_funcs]
        den_a = ti_d - ti
        den_b = ti_d1 - ti1
        safe_a = torch.where(den_a > 0, den_a, torch.ones_like(den_a))
        safe_b = torch.where(den_b > 0, den_b, torch.ones_like(den_b))
        a = torch.where(den_a > 0, (x[:, None] - ti) / safe_a, torch.zeros_like(x[:, None] - ti))
        b = torch.where(den_b > 0, (ti_d1 - x[:, None]) / safe_b, torch.zeros_like(x[:, None] - ti))
        basis = a * basis[:, :n_funcs] + b * basis[:, 1: n_funcs + 1]
    n_funcs = n_knots - 1 - k
    ti, ti_k = t[:n_funcs], t[k: k + n_funcs]
    ti1, ti_k1 = t[1: 1 + n_funcs], t[k + 1: k + 1 + n_funcs]
    den_a, den_b = ti_k - ti, ti_k1 - ti1
    inv_a = torch.where(den_a > 0, 1.0 / torch.where(den_a > 0, den_a, torch.ones_like(den_a)),
                        torch.zeros_like(den_a))
    inv_b = torch.where(den_b > 0, 1.0 / torch.where(den_b > 0, den_b, torch.ones_like(den_b)),
                        torch.zeros_like(den_b))
    derivative = k * (lower[:, :n_funcs] * inv_a - lower[:, 1: n_funcs + 1] * inv_b)
    return basis, derivative


class JawCalibrationMap(torch.nn.Module):
    """Smooth map between physical jaw coordinates and the FLAME jaw axis-angle."""

    def __init__(self, calibration: dict):
        super().__init__()
        q_open = np.asarray(calibration["grid_q_open_rad"], dtype=np.float64)
        q_lat = np.asarray(calibration["grid_q_lat_rad"], dtype=np.float64)
        theta = np.asarray(calibration["grid_theta_jaw"], dtype=np.float64)
        self.alpha_1 = float(calibration["alpha_1"])
        self.alpha_2 = float(calibration["alpha_2"])
        open_bins = np.unique(q_open)
        lateral_bins = np.unique(q_lat)
        grid = np.empty((len(open_bins), len(lateral_bins), 3))
        for qo, ql, value in zip(q_open, q_lat, theta, strict=True):
            i = int(np.flatnonzero(np.isclose(open_bins, qo, atol=1e-14))[0])
            j = int(np.flatnonzero(np.isclose(lateral_bins, ql, atol=1e-14))[0])
            grid[i, j] = value
        self.open_bins = open_bins
        self.lateral_bins = lateral_bins
        self.theta_grid = grid
        splines = [RectBivariateSpline(open_bins, lateral_bins, grid[:, :, axis], kx=3, ky=3, s=0.0)
                   for axis in range(3)]
        tx, ty = splines[0].tck[:2]
        coefficients = np.stack([s.tck[2].reshape(len(tx) - 4, len(ty) - 4) for s in splines])
        knots_open = torch.from_numpy(np.asarray(tx, dtype=np.float64))
        knots_lateral = torch.from_numpy(np.asarray(ty, dtype=np.float64))
        # Piecewise power form of the same bicubic spline: on every knot cell the
        # spline is a polynomial in the local coordinates (u, v) in [0, 1]; its 4 x 4
        # coefficients are recovered exactly from 16 samples of the B-spline form.
        cuts_open = torch.unique(knots_open)
        cuts_lateral = torch.unique(knots_lateral)
        nodes = torch.linspace(0.0, 1.0, 4, dtype=torch.float64)
        vandermonde = nodes[:, None] ** torch.arange(4, dtype=torch.float64)[None, :]
        inverse_v = torch.linalg.inv(vandermonde)
        coeff_t = torch.from_numpy(coefficients)
        cells = torch.empty(3, len(cuts_open) - 1, len(cuts_lateral) - 1, 4, 4, dtype=torch.float64)
        for i in range(len(cuts_open) - 1):
            xo = cuts_open[i] + nodes * (cuts_open[i + 1] - cuts_open[i])
            bo, _ = _bspline_basis(xo, knots_open, 3)
            for j in range(len(cuts_lateral) - 1):
                xl = cuts_lateral[j] + nodes * (cuts_lateral[j + 1] - cuts_lateral[j])
                bl, _ = _bspline_basis(xl, knots_lateral, 3)
                samples = torch.einsum("pi,aij,qj->apq", bo, coeff_t, bl)        # (3, 4, 4)
                cells[:, i, j] = inverse_v @ samples @ inverse_v.T
        self.register_buffer("cuts_open", cuts_open)
        self.register_buffer("cuts_lateral", cuts_lateral)
        self.register_buffer("cell_coefficients", cells)                         # (3, n_cells_o, n_cells_l, 4, 4)
        self.register_buffer("lower", torch.tensor([open_bins[0], lateral_bins[0]], dtype=torch.float64))
        self.register_buffer("upper", torch.tensor([open_bins[-1], lateral_bins[-1]], dtype=torch.float64))
        seed_open = np.linspace(open_bins[0], open_bins[-1], 21)
        seed_lateral = np.linspace(lateral_bins[0], lateral_bins[-1], 21)
        qo, ql = np.meshgrid(seed_open, seed_lateral, indexing="ij")
        self.register_buffer("seed_q", torch.from_numpy(np.stack([qo.ravel(), ql.ravel()], axis=1)))
        with torch.no_grad():
            self.register_buffer("seed_theta", self._evaluate(self.seed_q)[0])

    @property
    def q_open_max(self) -> float:
        return float(self.open_bins[-1])

    def _evaluate(self, q: torch.Tensor):
        """theta (B,3) and Jacobian d theta / d q (B,3,2) at q (B,2), float64."""
        co, cl = self.cuts_open, self.cuts_lateral
        i = (torch.searchsorted(co, q[:, 0].contiguous(), right=True) - 1).clamp(0, len(co) - 2)
        j = (torch.searchsorted(cl, q[:, 1].contiguous(), right=True) - 1).clamp(0, len(cl) - 2)
        ho = co[i + 1] - co[i]
        hl = cl[j + 1] - cl[j]
        u = (q[:, 0] - co[i]) / ho
        v = (q[:, 1] - cl[j]) / hl
        powers = torch.arange(4, dtype=q.dtype, device=q.device)
        pu = u[:, None] ** powers
        pv = v[:, None] ** powers
        du = torch.cat([torch.zeros_like(u[:, None]), pu[:, :3] * powers[1:]], dim=1)
        dv = torch.cat([torch.zeros_like(v[:, None]), pv[:, :3] * powers[1:]], dim=1)
        C = self.cell_coefficients[:, i, j].transpose(0, 1)                     # (B, 3, 4, 4)
        Cv = (C @ pv[:, None, :, None])[..., 0]                                 # (B, 3, 4)
        theta = (Cv @ pu[:, :, None])[..., 0]
        d_open = (Cv @ du[:, :, None])[..., 0] / ho[:, None]
        d_lateral = ((C @ dv[:, None, :, None])[..., 0] @ pu[:, :, None])[..., 0] / hl[:, None]
        return theta, torch.stack([d_open, d_lateral], dim=-1)

    def physical_to_theta(self, q_open: torch.Tensor, q_lateral: torch.Tensor) -> torch.Tensor:
        """FLAME jaw axis-angle (B,3) of physical coordinates (B,) each; differentiable.

        Inputs are clamped to the calibrated domain."""
        q = torch.stack([q_open, q_lateral], dim=-1).to(torch.float64)
        q = torch.maximum(torch.minimum(q, self.upper), self.lower)
        return self._evaluate(q)[0]

    # ------------------------------------------------------------------ projection (NumPy)
    # The projection is an iterative solve on a handful of numbers per pose; NumPy runs
    # it with microseconds of overhead per step where PyTorch spends close to a
    # millisecond per tiny operation. It is not differentiated through.

    def _numpy_tables(self):
        tables = self.__dict__.get("_np_tables")
        if tables is None:
            tables = {name: getattr(self, name).detach().cpu().numpy() for name in
                      ("cuts_open", "cuts_lateral", "cell_coefficients", "lower", "upper", "seed_q", "seed_theta")}
            self.__dict__["_np_tables"] = tables
        return tables

    def _evaluate_np(self, q: np.ndarray):
        t = self._numpy_tables()
        co, cl = t["cuts_open"], t["cuts_lateral"]
        i = np.clip(np.searchsorted(co, q[:, 0], side="right") - 1, 0, len(co) - 2)
        j = np.clip(np.searchsorted(cl, q[:, 1], side="right") - 1, 0, len(cl) - 2)
        ho, hl = co[i + 1] - co[i], cl[j + 1] - cl[j]
        u, v = (q[:, 0] - co[i]) / ho, (q[:, 1] - cl[j]) / hl
        powers = np.arange(4)
        pu, pv = u[:, None] ** powers, v[:, None] ** powers
        du = np.concatenate([np.zeros_like(u[:, None]), pu[:, :3] * powers[1:]], axis=1)
        dv = np.concatenate([np.zeros_like(v[:, None]), pv[:, :3] * powers[1:]], axis=1)
        C = np.moveaxis(t["cell_coefficients"][:, i, j], 0, 1)                   # (B, 3, 4, 4)
        Cv = np.einsum("bapq,bq->bap", C, pv)
        theta = np.einsum("bap,bp->ba", Cv, pu)
        d_open = np.einsum("bap,bp->ba", Cv, du) / ho[:, None]
        d_lateral = np.einsum("bapq,bp,bq->ba", C, pu, dv) / hl[:, None]
        return theta, np.stack([d_open, d_lateral], axis=-1)

    def _solve_np(self, q: np.ndarray, target: np.ndarray, iterations: int = 60) -> np.ndarray:
        """Bounded Levenberg-Marquardt from the seeds q (B,2)."""
        t = self._numpy_tables()
        lo, hi = t["lower"], t["upper"]
        lam = np.full(len(q), 1e-6)
        theta, jac = self._evaluate_np(q)
        residual = theta - target
        cost = (residual ** 2).sum(-1)
        for _ in range(iterations):
            g = np.einsum("bak,ba->bk", jac, residual)
            H = np.einsum("bak,bal->bkl", jac, jac)
            fixed = ((q <= lo + 1e-15) & (g > 0)) | ((q >= hi - 1e-15) & (g < 0))
            free = (~fixed).astype(float)
            Hf = H * free[:, :, None] * free[:, None, :] + np.eye(2) * (1 - free)[:, :, None]
            Hf = Hf + lam[:, None, None] * np.eye(2) * (np.diagonal(H, axis1=1, axis2=2)[:, :, None] + 1e-12)
            step = -_solve2_np(Hf, g * free)
            q_new = np.clip(q + step, lo, hi)
            theta_new, jac_new = self._evaluate_np(q_new)
            residual_new = theta_new - target
            cost_new = (residual_new ** 2).sum(-1)
            better = cost_new <= cost
            q = np.where(better[:, None], q_new, q)
            jac = np.where(better[:, None, None], jac_new, jac)
            residual = np.where(better[:, None], residual_new, residual)
            cost = np.where(better, cost_new, cost)
            lam = np.where(better, np.maximum(lam * 0.3, 1e-12), lam * 10.0)
            if bool(((np.abs(step).max(-1) < 1e-14) | (lam > 1e8)).all()):
                break
        return q

    def _project_np(self, target: np.ndarray):
        t = self._numpy_tables()
        lo, hi = t["lower"], t["upper"]
        diagonal = np.clip(np.stack([target[:, 0] / self.alpha_1, target[:, 1] / self.alpha_2], axis=-1), lo, hi)
        distance = ((t["seed_theta"][None, :, :] - target[:, None, :]) ** 2).sum(-1)
        nearest = t["seed_q"][distance.argmin(axis=1)]
        q1 = self._solve_np(diagonal, target)
        q2 = self._solve_np(nearest, target)
        c1 = ((self._evaluate_np(q1)[0] - target) ** 2).sum(-1)
        c2 = ((self._evaluate_np(q2)[0] - target) ** 2).sum(-1)
        q = np.where((c2 < c1)[:, None], q2, q1)
        theta, jac = self._evaluate_np(q)
        residual = theta - target
        # unconstrained Gauss-Newton step from the bounded optimum: how far the
        # boundary holds the projection back
        g = np.einsum("bak,ba->bk", jac, residual)
        H = np.einsum("bak,bal->bkl", jac, jac) + 1e-12 * np.eye(2)
        step = -_solve2_np(H, g)
        outward = ((q <= lo + 1e-12) & (step < -OUT_OF_DOMAIN_TOLERANCE_RAD)) | \
                  ((q >= hi - 1e-12) & (step > OUT_OF_DOMAIN_TOLERANCE_RAD))
        return q, ~outward.any(-1), np.linalg.norm(residual, axis=-1)

    @torch.no_grad()
    def project(self, theta_jaw: torch.Tensor):
        """Project FLAME jaw poses (B,3) onto the calibrated manifold (cached per distinct pose).

        Returns ``q`` (B,2) physical coordinates (float64), ``in_domain`` (B,) bool and
        the off-manifold ``residual`` (B,) in radians. A pose whose projection is held
        back by the domain boundary by more than half a degree is out of domain.
        """
        target = np.asarray(theta_jaw.detach().cpu().numpy(), dtype=np.float64).reshape(-1, 3)
        keys = [row.tobytes() for row in target]
        cache = self.__dict__.setdefault("_projection_cache", {})
        missing = sorted({k for k in keys if k not in cache})
        if missing:
            rows = np.frombuffer(b"".join(missing), dtype=np.float64).reshape(-1, 3)
            q, ok, res = self._project_np(rows)
            if len(cache) > 65536:
                cache.clear()
            for n, key in enumerate(missing):
                cache[key] = (q[n], ok[n], res[n])
        device = theta_jaw.device
        q = torch.from_numpy(np.stack([cache[k][0] for k in keys])).to(device)
        ok = torch.from_numpy(np.array([cache[k][1] for k in keys])).to(device)
        res = torch.from_numpy(np.array([cache[k][2] for k in keys])).to(device)
        return q, ok, res


def _solve2_np(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    det = A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] * A[:, 1, 0]
    return np.stack([(A[:, 1, 1] * b[:, 0] - A[:, 0, 1] * b[:, 1]) / det,
                     (A[:, 0, 0] * b[:, 1] - A[:, 1, 0] * b[:, 0]) / det], axis=-1)


class JawMuscleLaw:
    """The jaw muscle law for one muscle vocabulary (columns resolved by name)."""

    def __init__(self, muscle_names, q_open_max: float):
        names = [str(n) for n in muscle_names]
        missing = [n for n in OPENERS + MASSETER + TEMPORALIS if n not in names]
        if missing:
            raise ValueError(f"jaw muscle law: vocabulary lacks {missing}")
        self.q_open_max = float(q_open_max)
        self.openers = [names.index(n) for n in OPENERS]
        self.masseter = [names.index(n) for n in MASSETER]
        self.temporalis = [names.index(n) for n in TEMPORALIS]
        self.columns = sorted(self.openers + self.masseter + self.temporalis)
        self.kappa_masseter = KAPPA_MASSETER
        self.kappa_temporalis = KAPPA_TEMPORALIS

    def drive(self, activations: torch.Tensor) -> torch.Tensor:
        """Net opening drive in [0, 1], (B,)."""
        a = activations
        net = (a[:, self.openers].mean(-1)
               - self.kappa_masseter * a[:, self.masseter].mean(-1)
               - self.kappa_temporalis * a[:, self.temporalis].mean(-1))
        return net.clamp(0.0, 1.0)

    def q_open(self, activations: torch.Tensor) -> torch.Tensor:
        """Jaw opening (rad) produced by the activations, (B,)."""
        return self.q_open_max * self.drive(activations)

    def fill(self, activations: torch.Tensor, q_open: torch.Tensor) -> torch.Tensor:
        """Copy of ``activations`` with the six jaw columns set to the minimum-effort inverse of q_open."""
        drive = (q_open.to(torch.float64) / self.q_open_max).clamp(0.0, 1.0).to(activations.dtype)
        out = activations.clone()
        out[:, self.masseter + self.temporalis] = 0.0
        out[:, self.openers] = drive[:, None].expand(-1, len(self.openers))
        return out
