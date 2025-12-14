import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_88a62173(I: Grid) -> Grid:
    x0 = lefthalf(I)
    x1 = righthalf(I)
    x2 = tophalf(x0)
    x3 = tophalf(x1)
    x4 = bottomhalf(x0)
    x5 = bottomhalf(x1)
    x6 = astuple(x2, x3)
    x7 = astuple(x4, x5)
    x8 = combine(x6, x7)
    x9 = leastcommon(x8)
    return x9