import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_72ca375d(I: Grid) -> Grid:
    x0 = objects(I, T, T, T)
    x1 = fork(equality, identity, vmirror)
    x2 = extract(x0, x1)
    x3 = subgrid(x2, I)
    return x3