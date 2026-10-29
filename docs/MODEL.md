# The MyoFLAME model

## What goes in and what comes out

MyoFLAME takes the parameters of FLAME 2020 (identity `betas`, expression `expression`,
and the five joints `global_orient`, `neck_pose`, `jaw_pose`, `leye_pose`, `reye_pose`)
and returns

* `activations` (B, 50): how strongly each of 50 named muscle columns is contracted, in [0, 1];
* `vertices` (B, 5023, 3): the skin those muscles produce, in FLAME topology;
* `jaw_open`, `jaw_lateral` (B,): the physical jaw coordinates in radians.

The identity, head, neck, eyes and jaw joint are FLAME's own. Only the expression
blendshapes are replaced by the muscle response:

```
v = FLAME(beta, expression = 0, pose) + T(pose) R(q_open, q_lateral) a
```

* `R(q)` is the muscle response basis: for every muscle column the displacement of
  3,931 FLAME vertices (all vertices except the eyeballs) per unit activation, stored on
  a 5 x 5 grid of jaw positions (opening 0 to 20 degrees, lateral excursion -12 to 12
  degrees) and bilinearly interpolated in between.
* `T(pose)` carries those displacements from the jaw-only pose into the full head, neck
  and eye pose (the linear part of FLAME's skinning).
* `a` is either predicted by the activation surrogate (default, batched, differentiable)
  or solved exactly by inverse physics (`activation_mode="exact"`, see below).

Because the topology is FLAME's, FLAME landmark embeddings, vertex masks, UV maps and
renderers apply to MyoFLAME meshes unchanged.

## The 50 muscle columns

Bilateral muscles have one column per side (`_L`, `_R`; the right column mirrors the left
one). Line muscles pull a footprint of skin from their skin insertion towards their bone
origin. Ring muscles contract a ring of skin: the orbicularis oris in four quadrants
(upper and lower lip, each side) and the orbicularis oculi in an orbital part (around the
eye) and a palpebral part (the lids). The jaw muscles move the mandible (next section).
The action units are the FACS codes whose standard muscle list contains the muscle
(`myoflame.load_action_units()`).

| columns | muscle | label | type | FACS action units |
|---|---|---|---|---|
| 0, 1 | Buccinator | Bucc | line muscle | AU14 |
| 2, 3 | Corrugator Supercilii | CorSup | line muscle | AU4, AU44 |
| 4, 5 | Depressor Anguli Oris | DAO | line muscle | AU15 |
| 6, 7 | Depressor Labii Inferioris | DLI | line muscle | AU16, AU25 |
| 8, 9 | Depressor Supercilii | DepSup | line muscle | AU4 |
| 10, 11 | Digastric | Diga | jaw muscle | AU26, AU27 |
| 12, 13 | Frontalis Inner | FrontInn | line muscle | AU1 |
| 14, 15 | Frontalis Outer | FrontOut | line muscle | AU2 |
| 16, 17 | Levator Anguli Oris | LAO | line muscle | AU13 |
| 18, 19 | Levator Labii Superioris Alaeque Nasi | LLSAN | line muscle | AU9 |
| 20, 21 | Levator Labii Superioris | LLS | line muscle | AU10 |
| 22, 23 | Masseter | Mass | jaw muscle | - |
| 24, 25 | Mentalis | Ment | line muscle | AU17 |
| 26, 27 | Orbicularis Oculi Orbital | OOculOrb | ring muscle | AU6, AU44, AU46 |
| 28, 29 | Procerus | Proc | line muscle | AU4, AU9 |
| 30, 31 | Risorius | Riso | line muscle | AU20 |
| 32, 33 | Temporalis | Temp | jaw muscle | - |
| 34, 35 | Transverse Nasalis | TraNas | line muscle | AU38, AU39 |
| 36, 37 | Zygomaticus Major | ZygMaj | line muscle | AU12 |
| 38, 39 | Zygomaticus Minor | ZygMin | line muscle | AU11 |
| 40, 41 | Orbicularis Oris upper | OOrisU | ring muscle | AU18, AU22, AU23, AU24, AU28 |
| 42, 43 | Orbicularis Oris lower | OOrisL | ring muscle | AU18, AU22, AU23, AU24, AU28 |
| 44, 45 | Orbicularis Oculi Palpebral | OOculPal | ring muscle | AU7, AU43, AU44, AU45, AU46 |
| 46, 47 | Levator Palpebrae Superioris | LPS | line muscle | AU5 |
| 48, 49 | Platysma | Plat | line muscle | AU20, AU21 |

## The jaw muscles

The six jaw columns (digastric, masseter, temporalis, each side) set the jaw opening
through one invertible law:

```
q_open = 20 deg * clip(mean(a_digastric) - 5.23 mean(a_masseter) - 2.10 mean(a_temporalis), 0, 1)
```

The digastric columns stand for the jaw-depressor group; 1.0 on both sides opens the jaw
to the calibrated maximum of 20 degrees. The closers enter with their torque capacity
relative to the depressors (force times moment arm: masseter 8.60 N m, temporalis
3.45 N m, digastric 1.65 N m), so a fifth of the masseter holds the jaw shut against a
fully active depressor group. When MyoFLAME reads a FLAME jaw pose it reports the
minimum-effort activations of that opening: digastric = opening / 20 degrees on both
sides, closers at zero. Lateral excursion is a pose input (`jaw_lateral`), not a muscle.

The opening moves the skin through FLAME's own jaw joint. A calibration fitted FLAME's
jaw rotation to a rigid mandible rotating about the temporomandibular joints on a 5 x 5
grid of (opening, lateral) positions; three bicubic splines interpolate FLAME's jaw
axis-angle over that grid (`model.jaw_map.physical_to_theta`). A FLAME jaw pose is mapped
back to physical coordinates by projecting it onto this surface (bounded least squares,
`model.jaw_map.project`). This projection is not differentiated through: gradients with
respect to `jaw_pose` reach the skin through FLAME's jaw joint but not the activations or the
interpolation of the response basis. In `from_activations` the opening is a differentiable
function of the jaw activations. Jaw poses outside the calibrated range are reported in
`jaw_in_domain`; the skin keeps the given jaw rotation, the activations use the nearest
calibrated position.

## Where the activations come from

**Surrogate (default).** A small network predicts the activations from the FLAME
parameters:

```
a = clip(W(q) expression + g(|expression|) alpha * tanh(MLP(betas, expression, q)), 0, 1)
```

`W(q)` is a linear prior per jaw cell, the MLP a residual (403 -> 256 -> 256 -> 128 -> 50),
and `g` fades the residual out towards the neutral face, so a zero expression gives zero
activations and MyoFLAME's skin equals FLAME's exactly. It was trained to reproduce the
exact inverse-physics activations of the 218 FACS-coded D3DFACS expressions on 1,211 LYHM
identities, so it is most reliable for expressions of that kind.

**Exact inverse physics** (`activation_mode="exact"`). For each sample MyoFLAME solves

```
min_a || R(q) a - d ||^2 + 1e-4 || diag(s) a ||^2    subject to  0 <= a <= 1
```

where `d` is FLAME's expression displacement at the observed vertices and `s` the column
norms. It needs no training and accepts any FLAME 2020 expression, at the cost of one
bounded least-squares solve per sample on the CPU; it is not differentiable.

## Going back to FLAME

`model.to_flame(output)` finds the FLAME expression coefficients whose skin is closest to
MyoFLAME's (least squares over all vertices) and returns them with the jaw pose, so
muscle-driven results can be passed to tools that only understand FLAME parameters. The
fit is not exact: muscle-driven skin is not confined to FLAME's expression space, and the
returned `rms` says how far it is.

## Files

| file | contents |
|---|---|
| `myoflame/data/muscle_responses.npz` | `R_grid` (5, 5, 11793, 50) float32, jaw bins, observed vertex indices, muscle names and types |
| `myoflame/data/surrogate.pt` | surrogate weights (PyTorch state dict) |
| `myoflame/data/jaw_calibration.npz` | jaw calibration grid and temporomandibular joint frame |
| `myoflame/data/expressions.npz` | the 218 D3DFACS expressions as FLAME 2020 parameters (`label` is the FACS code) |
| `myoflame/data/action_units.json` | FACS action unit to muscle columns |
| `viewer/data/*.bin` | the same model, in float16, for the browser viewer |
| `viewer/data/flame.bin`, `verify.bin` | FLAME 2020 and reference meshes for the viewer, written by `demos/build_viewer_data.py` from your own FLAME file (not in the repository) |
