import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_46f33fce(I: Grid) -> Grid:
    x0 = rot180(I)
    x1 = downscale(x0, TWO)
    x2 = rot180(x1)
    x3 = upscale(x2, FOUR)
    return x3