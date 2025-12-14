import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_f25fbde4(I: Grid) -> Grid:
    x0 = compress(I)
    x1 = upscale(x0, TWO)
    return x1