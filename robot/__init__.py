"""robot — Robot description loaders (URDF, manual builders)."""

from .model import RobotModel
from .urdf_loader import load_urdf

__all__ = ["RobotModel", "load_urdf"]
