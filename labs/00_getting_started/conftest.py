"""Ensure lab tests import the local analysis module."""

from __future__ import annotations

import sys
from pathlib import Path

_LAB_DIR = Path(__file__).resolve().parent
if str(_LAB_DIR) not in sys.path:
    sys.path.insert(0, str(_LAB_DIR))
