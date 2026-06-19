"""SD-MSCF-DET research fork of Ultralytics."""

import os

__version__ = "8.3.163-sd-mscf-det"

if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = "1"

from ultralytics.models.yolo.model import YOLOMM

__all__ = ("YOLOMM", "__version__")
