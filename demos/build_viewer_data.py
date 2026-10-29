"""Write the FLAME part of the viewer's data from your own FLAME 2020 file.

FLAME cannot be redistributed, so the viewer's FLAME arrays are not part of this
repository. This script writes them, once, into viewer/data/:

    flame.bin    FLAME 2020 template, faces, the first ten identity and all hundred
                 expression blend shapes, pose blend shapes, skinning weights and joint
                 regressor, in the layout given by viewer/data/manifest.json
    verify.bin   reference meshes computed with the Python model, against which the
                 viewer checks its browser port on load

    python demos/build_viewer_data.py [--flame model/FLAME2020/generic_model.pkl]

The files are git-ignored. Do not publish them: they carry FLAME 2020, which keeps
its own licence.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from myoflame import MyoFLAME  # noqa: E402
from myoflame.flame import load_flame  # noqa: E402

DATA = ROOT / "viewer" / "data"
DEFAULT_FLAME = ROOT / "model" / "FLAME2020" / "generic_model.pkl"


def _write(entry: dict, arrays: dict, folder: Path) -> None:
    """Write one bundle at the offsets recorded in the manifest."""
    buffer = bytearray(entry["nbytes"])
    for name, spec in entry["arrays"].items():
        value = np.ascontiguousarray(np.asarray(arrays[name]).astype(spec["dtype"]))
        if list(value.shape) != spec["shape"] or value.nbytes != spec["nbytes"]:
            raise ValueError(f"{entry['file']}:{name} has shape {value.shape}, expected {spec['shape']}")
        buffer[spec["offset"]:spec["offset"] + spec["nbytes"]] = value.tobytes()
    (folder / entry["file"]).write_bytes(bytes(buffer))


def flame_arrays(flame_path: Path) -> dict:
    flame = load_flame(flame_path)               # float32 arrays, shape and FLAME 2020 version checks
    return {
        "v_template": flame["v_template"],
        "faces": flame["faces"],
        "shapedirs_id10": flame["shapedirs"][:, :, :10],
        "shapedirs_expr": flame["shapedirs"][:, :, 300:400],
        "posedirs": flame["posedirs"],
        "weights": flame["weights"],
        "J_regressor": flame["J_regressor"],
    }


@torch.no_grad()
def reference_meshes(model: MyoFLAME, checks: list) -> dict:
    """The viewer's reference cases (manifest meta.checks) through the Python model."""
    out = {}
    for c in checks:
        betas = torch.zeros(1, 300)
        betas[0, :10] = torch.tensor(c["beta10"])
        psi = torch.tensor(c["psi"], dtype=torch.float32)[None]
        q_open = torch.tensor([c["q_open"]], dtype=torch.float64)
        q_lateral = torch.tensor([c["q_lat"]], dtype=torch.float64)
        pose = torch.zeros(1, 15)
        pose[0, 3:6] = torch.tensor(c["neck"])
        pose[0, 6:9] = model.jaw_map.physical_to_theta(q_open, q_lateral).float()
        if c["a_manual"] is None:
            a = model.surrogate(betas, psi, q_open.float(), q_lateral.float())
            a = model.jaw_law.fill(a, q_open)
        else:
            a = torch.tensor(c["a_manual"], dtype=torch.float32)[None]
            if not c["jaw_from_law"]:
                a = a * model.skin_mask
        out[c["name"] + "_myo"] = model._skin(betas, pose, None, a, q_open, q_lateral)[0].numpy()
        out[c["name"] + "_flame"] = model.flame(betas, psi, pose)[0].numpy()
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--flame", type=Path, default=DEFAULT_FLAME, help="FLAME 2020 generic_model.pkl")
    parser.add_argument("--out", type=Path, default=DATA, help="viewer data folder (default viewer/data)")
    args = parser.parse_args(argv)
    manifest = json.loads((DATA / "manifest.json").read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    _write(manifest["files"]["flame"], flame_arrays(args.flame), args.out)
    model = MyoFLAME(args.flame)
    _write(manifest["files"]["verify"], reference_meshes(model, manifest["meta"]["checks"]), args.out)
    print(f"wrote {manifest['files']['flame']['file']} and {manifest['files']['verify']['file']} to {args.out}")


if __name__ == "__main__":
    main()
