"""FLAME parameters in, muscle activations and muscle-driven skin out.

Runs FACS-coded D3DFACS expressions (shipped as FLAME 2020 parameters) through
MyoFLAME and writes, per expression, the MyoFLAME mesh, the plain FLAME mesh for
comparison, and the estimated muscle activations.

    python demos/demo_forward.py --out output/forward
    python demos/demo_forward.py --labels 12 1+2 27 --identity random --seed 3
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from myoflame import MyoFLAME, load_expressions, save_obj  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--flame", default="model/FLAME2020/generic_model.pkl", help="FLAME 2020 generic_model.pkl")
    parser.add_argument("--out", default="output/forward")
    parser.add_argument("--labels", nargs="+", default=["1+2", "4", "9", "12", "15", "18", "20", "27"],
                        help="FACS codes of D3DFACS expressions (see myoflame.load_expressions()['label'])")
    parser.add_argument("--identity", choices=["mean", "random"], default="mean")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exact", action="store_true", help="exact inverse physics instead of the surrogate")
    args = parser.parse_args(argv)

    model = MyoFLAME(args.flame)
    expressions = load_expressions()
    labels = [str(x) for x in expressions["label"]]
    missing = [label for label in args.labels if label not in labels]
    if missing:
        parser.error(f"unknown expression labels {missing}")
    index = [labels.index(label) for label in args.labels]
    B = len(index)
    betas = torch.zeros(B, 300)
    if args.identity == "random":
        generator = torch.Generator().manual_seed(args.seed)
        betas[:, :20] = torch.randn(1, 20, generator=generator).expand(B, -1)
    expression = torch.from_numpy(expressions["expression"][index])
    jaw_pose = torch.from_numpy(expressions["jaw_pose"][index])

    with torch.no_grad():
        out = model(betas=betas, expression=expression, jaw_pose=jaw_pose, return_flame=True,
                    activation_mode="exact" if args.exact else "surrogate")
        neutral = model(betas=betas[:1])

    folder = Path(args.out)
    folder.mkdir(parents=True, exist_ok=True)
    save_obj(folder / "neutral.obj", neutral.vertices[0], model.faces)
    summary = []
    for b, label in enumerate(args.labels):
        stem = label.replace("+", "_")
        save_obj(folder / f"{stem}_myoflame.obj", out.vertices[b], model.faces)
        save_obj(folder / f"{stem}_flame.obj", out.flame_vertices[b], model.faces)
        a = out.activations[b]
        top = torch.argsort(a, descending=True)[:5]
        gap = (out.vertices[b] - out.flame_vertices[b]).norm(dim=-1)
        summary.append({
            "label": label,
            "name": str(expressions["name"][index[b]]),
            "jaw_open_deg": round(float(torch.rad2deg(out.jaw_open[b])), 2),
            "strongest_muscles": {model.muscle_names[k]: round(float(a[k]), 3) for k in top if a[k] > 0.05},
            "distance_to_flame_mm": {"mean": round(float(gap.mean() * 1000), 3), "max": round(float(gap.max() * 1000), 3)},
        })
    with (folder / "activations.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label"] + model.vocabulary.labels)
        for b, label in enumerate(args.labels):
            writer.writerow([label] + [f"{x:.4f}" for x in out.activations[b].tolist()])
    (folder / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    for row in summary:
        muscles = ", ".join(f"{k} {v:.2f}" for k, v in row["strongest_muscles"].items())
        print(f"AU {row['label']:<8} jaw {row['jaw_open_deg']:5.1f} deg  {muscles}")
    print(f"wrote {2 * B + 1} meshes, activations.csv and summary.json to {folder}")
    return np.asarray(out.activations)


if __name__ == "__main__":
    main()
