import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_50cb2852(I: Grid) -> Grid:
    x0 = objects(I, T, F, T)
    x1 = compose(backdrop, inbox)
    x2 = mapply(x1, x0)
    x3 = fill(I, EIGHT, x2)
    return x3