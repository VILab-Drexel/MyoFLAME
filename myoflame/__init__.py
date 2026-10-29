"""MyoFLAME: muscle activations and muscle-driven skin for FLAME."""
from .io import load_action_units, load_expressions, save_obj
from .jaw import JawCalibrationMap, JawMuscleLaw
from .model import MyoFLAME, MyoFLAMEOutput

__version__ = "1.0.0"
__all__ = ["MyoFLAME", "MyoFLAMEOutput", "JawCalibrationMap", "JawMuscleLaw", "load_expressions",
           "load_action_units", "save_obj", "__version__"]
