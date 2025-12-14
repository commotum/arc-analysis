import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_ac0a08a4(I: Grid) -> Grid:
    x0 = mostcolor(I)
    x1 = colorcount(I, x0)
    x2 = height(I)
    x3 = width(I)
    x4 = multiply(x2, x3)
    x5 = subtract(x4, x1)
    x6 = upscale(I, x5)
    return x6