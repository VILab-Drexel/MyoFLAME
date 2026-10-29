# FLAME 2020

MyoFLAME runs on top of the FLAME 2020 head model, which cannot be redistributed.

1. Register at https://flame.is.tue.mpg.de and agree to the FLAME model licence.
2. Download **FLAME 2020** (`FLAME2020.zip`).
3. Copy `generic_model.pkl` from the archive into this folder:

```
model/FLAME2020/generic_model.pkl
```

Optionally also copy `FLAME_masks.pkl` (the FLAME vertex masks) next to it.

MyoFLAME reads the pickle without chumpy. It refuses other FLAME versions (for
example FLAME 2023), whose expression basis differs from the one MyoFLAME was
built on.
