from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / 'benchmarks'
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))
