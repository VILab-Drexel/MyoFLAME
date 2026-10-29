"""Drive the face with muscles directly: one muscle (or a pair) at a time.

Every group is swept from rest to full activation and written as an OBJ
sequence. The digastric opens the jaw; the masseter closes it again.

    python demos/demo_muscles.py --out output/muscles
    python demos/demo_muscles.py --muscles Zygomaticus_Major_L Digastric_L,Digastric_R --steps 9
"""
import argparse
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from myoflame import MyoFLAME, save_obj  # noqa: E402

DEFAULT_GROUPS = [
    "Zygomaticus_Major_L,Zygomaticus_Major_R",
    "Zygomaticus_Major_L",
    "Frontalis_Inner_L,Frontalis_Inner_R",
    "Corrugator_Supercilii_L,Corrugator_Supercilii_R",
    "Levator_Labii_Superioris_Alaeque_Nasi_L,Levator_Labii_Superioris_Alaeque_Nasi_R",
    "Depressor_Anguli_Oris_L,Depressor_Anguli_Oris_R",
    "Orbicularis_Oris_upper_L,Orbicularis_Oris_upper_R,Orbicularis_Oris_lower_L,Orbicularis_Oris_lower_R",
    "Orbicularis_Oculi_Palpebral_L,Orbicularis_Oculi_Palpebral_R",
    "Digastric_L,Digastric_R",
]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--flame", default="model/FLAME2020/generic_model.pkl")
    parser.add_argument("--out", default="output/muscles")
    parser.add_argument("--muscles", nargs="+", default=DEFAULT_GROUPS,
                        help="groups of comma-separated muscle names (see MyoFLAME(...).muscle_names)")
    parser.add_argument("--steps", type=int, default=6)
    args = parser.parse_args(argv)

    model = MyoFLAME(args.flame)
    names = model.muscle_names
    levels = torch.linspace(0.0, 1.0, args.steps)
    folder = Path(args.out)
    rows = []
    with torch.no_grad():
        sequences = []
        for group in args.muscles:
            columns = [names.index(n) for n in group.split(",")]
            a = torch.zeros(args.steps, len(names))
            a[:, columns] = levels[:, None]
            sequences.append((group.split(",")[0].removesuffix("_L") + ("" if len(columns) == 1 else "_group"), a))
        # the closers: masseter 0 to 0.2 against digastric 0.7
        a = torch.zeros(args.steps, len(names))
        a[:, [names.index("Digastric_L"), names.index("Digastric_R")]] = 0.7
        a[:, [names.index("Masseter_L"), names.index("Masseter_R")]] = torch.linspace(0.0, 0.2, args.steps)[:, None]
        sequences.append(("Masseter_closing", a))
        for stem, a in sequences:
            out = model.from_activations(a)
            target = folder / stem
            target.mkdir(parents=True, exist_ok=True)
            for k in range(args.steps):
                save_obj(target / f"frame_{k:02d}.obj", out.vertices[k], model.faces)
                rows.append([stem, k, f"{float(a[k].max()):.3f}", f"{float(torch.rad2deg(out.jaw_open[k])):.2f}"])
            print(f"{stem:<36} jaw opening {torch.rad2deg(out.jaw_open).numpy().round(1).tolist()} deg")
    with (folder / "jaw.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sequence", "frame", "max_activation", "jaw_open_deg"])
        writer.writerows(rows)
    print(f"wrote {len(rows)} meshes to {folder}")


if __name__ == "__main__":
    main()
