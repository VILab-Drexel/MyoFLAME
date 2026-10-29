"""The 50 named muscle columns: names, types, bilateral pairs, short labels.

Type A: line muscles (bone origin to skin insertion). Type B: ring muscles
(orbicularis oris in four quadrants, orbicularis oculi orbital and palpebral
parts). Type C: jaw muscles (digastric, masseter, temporalis), which move the
mandible through the jaw muscle law instead of acting on the skin directly.

Naming: ``_L`` / ``_R`` is the side (the right column mirrors the left one);
``_upper`` / ``_lower`` and ``_Orbital`` / ``_Palpebral`` name parts of a ring.
"""
from __future__ import annotations

import re

ABBREVIATIONS = {
    "Buccinator": "Bucc", "Corrugator_Supercilii": "CorSup", "Depressor_Anguli_Oris": "DAO",
    "Depressor_Labii_Inferioris": "DLI", "Depressor_Supercilii": "DepSup", "Digastric": "Diga",
    "Frontalis_Inner": "FrontInn", "Frontalis_Outer": "FrontOut", "Levator_Anguli_Oris": "LAO",
    "Levator_Labii_Superioris_Alaeque_Nasi": "LLSAN", "Levator_Labii_Superioris": "LLS",
    "Masseter": "Mass", "Mentalis": "Ment", "Orbicularis_Oculi_Orbital": "OOculOrb",
    "Procerus": "Proc", "Risorius": "Riso", "Temporalis": "Temp", "Transverse_Nasalis": "TraNas",
    "Zygomaticus_Major": "ZygMaj", "Zygomaticus_Minor": "ZygMin", "Orbicularis_Oris_upper": "OOrisU",
    "Orbicularis_Oris_lower": "OOrisL", "Orbicularis_Oculi_Palpebral": "OOculPal",
    "Levator_Palpebrae_Superioris": "LPS", "Platysma": "Plat",
}

TYPE_DESCRIPTION = {"A": "line muscle", "B": "ring muscle", "C": "jaw muscle"}


def side_of(name: str) -> str | None:
    if name.endswith("_L"):
        return "L"
    if name.endswith("_R"):
        return "R"
    return None


def base_name(name: str) -> str:
    """Name without its side suffix."""
    return re.sub(r"_(L|R)$", "", name)


def mirror_name(name: str) -> str:
    """The contralateral column name (identity for midline columns)."""
    if name.endswith("_L"):
        return name[:-2] + "_R"
    if name.endswith("_R"):
        return name[:-2] + "_L"
    return name


def short_label(name: str) -> str:
    """Compact label such as ``ZygMaj L``."""
    side = side_of(name)
    abbreviation = ABBREVIATIONS.get(base_name(name), base_name(name))
    return abbreviation if side is None else f"{abbreviation} {side}"


class Vocabulary:
    """Column vocabulary (immutable view on names and types)."""

    def __init__(self, names, types):
        self.names = [str(n) for n in names]
        self.types = [str(t) for t in types]
        if len(self.names) != len(self.types) or len(set(self.names)) != len(self.names):
            raise ValueError("inconsistent muscle vocabulary")
        self.jaw = [k for k, t in enumerate(self.types) if t == "C"]
        self.soft_tissue = [k for k, t in enumerate(self.types) if t != "C"]
        self.pairs = [(k, self.names.index(mirror_name(n))) for k, n in enumerate(self.names)
                      if n.endswith("_L") and mirror_name(n) in self.names]
        self.labels = [short_label(n) for n in self.names]

    def __len__(self) -> int:
        return len(self.names)

    def index(self, name: str) -> int:
        return self.names.index(name)

    def mirror_index(self, k: int) -> int:
        return self.names.index(mirror_name(self.names[k]))
