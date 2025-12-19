import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_b91ae062(I: Grid) -> Grid:
    x0 = numcolors(I)
    x1 = decrement(x0)
    x2 = upscale(I, x1)
    return x2