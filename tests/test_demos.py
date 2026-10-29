"""Demo smoke tests. Need FLAME 2020 in model/FLAME2020/ and skip otherwise."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FLAME = ROOT / "model" / "FLAME2020" / "generic_model.pkl"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "demos" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(FLAME.is_file(), "FLAME 2020 not found in model/FLAME2020/")
class Demos(unittest.TestCase):
    def test_forward(self):
        with tempfile.TemporaryDirectory() as folder:
            load("demo_forward").main(["--flame", str(FLAME), "--out", folder, "--labels", "12", "27"])
            meshes = sorted(Path(folder).glob("*.obj"))
            self.assertEqual(len(meshes), 5)
            vertices = [line for line in meshes[0].read_text().splitlines() if line.startswith("v ")]
            self.assertEqual(len(vertices), 5023)

    def test_muscles(self):
        with tempfile.TemporaryDirectory() as folder:
            load("demo_muscles").main(["--flame", str(FLAME), "--out", folder, "--steps", "3",
                                       "--muscles", "Digastric_L,Digastric_R"])
            rows = (Path(folder) / "jaw.csv").read_text().splitlines()
            self.assertEqual(rows[3].split(",")[-1], "20.00")

    def test_viewer_data(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            load("build_viewer_data").main(["--flame", str(FLAME), "--out", folder])
            manifest = json.loads((ROOT / "viewer" / "data" / "manifest.json").read_text())
            for key in ("flame", "verify"):
                entry = manifest["files"][key]
                self.assertEqual((Path(folder) / entry["file"]).stat().st_size, entry["nbytes"])


if __name__ == "__main__":
    unittest.main()
