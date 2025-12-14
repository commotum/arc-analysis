import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_0d3d703e(I: Grid) -> Grid:
    x0 = switch(I, THREE, FOUR)
    x1 = switch(x0, EIGHT, NINE)
    x2 = switch(x1, TWO, SIX)
    x3 = switch(x2, ONE, FIVE)
    return x3