import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_44f52bb0(I: Grid) -> Grid:
    x0 = vmirror(I)
    x1 = equality(x0, I)
    x2 = hmirror(I)
    x3 = equality(x2, I)
    x4 = either(x1, x3)
    x5 = branch(x4, ONE, SEVEN)
    x6 = canvas(x5, UNITY)
    return x6