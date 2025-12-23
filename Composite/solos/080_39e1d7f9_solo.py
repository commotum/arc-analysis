# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "39e1d7f9"
SERIAL = "080"
URL    = "https://arcprize.org/play?task=39e1d7f9"

# --- Code Golf Concepts ---
CONCEPTS = [
    "detect_grid",
    "pattern_repetition",
    "grid_coloring",
]

# --- Example Grids ---
E1_IN = np.array([
    [0, 0, 0, 0, 8, 3, 3, 3, 3, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 3, 3, 3, 3, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 3, 3, 3, 3, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 3, 3, 3, 3, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [3, 3, 3, 3, 8, 6, 6, 6, 6, 8, 3, 3, 3, 3, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [3, 3, 3, 3, 8, 6, 6, 6, 6, 8, 3, 3, 3, 3, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [3, 3, 3, 3, 8, 6, 6, 6, 6, 8, 3, 3, 3, 3, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [3, 3, 3, 3, 8, 6, 6, 6, 6, 8, 3, 3, 3, 3, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 0, 0, 8, 3, 3, 3, 3, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 3, 3, 3, 3, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 3, 3, 3, 3, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 3, 3, 3, 3, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 6, 6, 6, 6, 8, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 6, 6, 6, 6, 8, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 6, 6, 6, 6, 8, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 6, 6, 6, 6, 8, 0, 0, 0, 0],
], dtype=int)

E1_OUT = np.array([
    [0, 0, 0, 0, 8, 3, 3, 3, 3, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 3, 3, 3, 3, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 3, 3, 3, 3, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 3, 3, 3, 3, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [3, 3, 3, 3, 8, 6, 6, 6, 6, 8, 3, 3, 3, 3, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [3, 3, 3, 3, 8, 6, 6, 6, 6, 8, 3, 3, 3, 3, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [3, 3, 3, 3, 8, 6, 6, 6, 6, 8, 3, 3, 3, 3, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [3, 3, 3, 3, 8, 6, 6, 6, 6, 8, 3, 3, 3, 3, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 0, 0, 8, 3, 3, 3, 3, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 3, 3, 3, 3, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 3, 3, 3, 3, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 3, 3, 3, 3, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 3, 3, 3, 3, 8, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 3, 3, 3, 3, 8, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 3, 3, 3, 3, 8, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 3, 3, 3, 3, 8, 0, 0, 0, 0],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 3, 3, 3, 3, 8, 6, 6, 6, 6, 8, 3, 3, 3, 3],
    [0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 3, 3, 3, 3, 8, 6, 6, 6, 6, 8, 3, 3, 3, 3],
    [0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 3, 3, 3, 3, 8, 6, 6, 6, 6, 8, 3, 3, 3, 3],
    [0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 3, 3, 3, 3, 8, 6, 6, 6, 6, 8, 3, 3, 3, 3],
], dtype=int)

E2_IN = np.array([
    [0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 4, 4, 4],
    [0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 4, 4, 4],
    [0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 4, 4, 4],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [0, 0, 0, 3, 0, 0, 0, 3, 6, 6, 6, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0],
    [0, 0, 0, 3, 0, 0, 0, 3, 6, 6, 6, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0],
    [0, 0, 0, 3, 0, 0, 0, 3, 6, 6, 6, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [0, 0, 0, 3, 6, 6, 6, 3, 4, 4, 4, 3, 6, 6, 6, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0],
    [0, 0, 0, 3, 6, 6, 6, 3, 4, 4, 4, 3, 6, 6, 6, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0],
    [0, 0, 0, 3, 6, 6, 6, 3, 4, 4, 4, 3, 6, 6, 6, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [0, 0, 0, 3, 0, 0, 0, 3, 6, 6, 6, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0],
    [0, 0, 0, 3, 0, 0, 0, 3, 6, 6, 6, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0],
    [0, 0, 0, 3, 0, 0, 0, 3, 6, 6, 6, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0],
    [0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0],
    [0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 4, 4, 4, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0],
    [0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 4, 4, 4, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0],
    [0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 4, 4, 4, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0],
    [0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0],
    [0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0],
], dtype=int)

E2_OUT = np.array([
    [0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 6, 6, 6, 3, 4, 4, 4],
    [0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 6, 6, 6, 3, 4, 4, 4],
    [0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 6, 6, 6, 3, 4, 4, 4],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [0, 0, 0, 3, 0, 0, 0, 3, 6, 6, 6, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 6, 6, 6],
    [0, 0, 0, 3, 0, 0, 0, 3, 6, 6, 6, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 6, 6, 6],
    [0, 0, 0, 3, 0, 0, 0, 3, 6, 6, 6, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 6, 6, 6],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [0, 0, 0, 3, 6, 6, 6, 3, 4, 4, 4, 3, 6, 6, 6, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0],
    [0, 0, 0, 3, 6, 6, 6, 3, 4, 4, 4, 3, 6, 6, 6, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0],
    [0, 0, 0, 3, 6, 6, 6, 3, 4, 4, 4, 3, 6, 6, 6, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [0, 0, 0, 3, 0, 0, 0, 3, 6, 6, 6, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0],
    [0, 0, 0, 3, 0, 0, 0, 3, 6, 6, 6, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0],
    [0, 0, 0, 3, 0, 0, 0, 3, 6, 6, 6, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 6, 6, 6, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0],
    [0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 6, 6, 6, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0],
    [0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 6, 6, 6, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [0, 0, 0, 3, 0, 0, 0, 3, 6, 6, 6, 3, 4, 4, 4, 3, 6, 6, 6, 3, 0, 0, 0, 3, 0, 0, 0],
    [0, 0, 0, 3, 0, 0, 0, 3, 6, 6, 6, 3, 4, 4, 4, 3, 6, 6, 6, 3, 0, 0, 0, 3, 0, 0, 0],
    [0, 0, 0, 3, 0, 0, 0, 3, 6, 6, 6, 3, 4, 4, 4, 3, 6, 6, 6, 3, 0, 0, 0, 3, 0, 0, 0],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 6, 6, 6, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0],
    [0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 6, 6, 6, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0],
    [0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 6, 6, 6, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0],
], dtype=int)

E3_IN = np.array([
    [0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 2, 2, 2],
    [0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 2, 2, 2],
    [0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 2, 2, 2],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 0, 8, 2, 2, 2, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0],
    [0, 0, 0, 8, 2, 2, 2, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0],
    [0, 0, 0, 8, 2, 2, 2, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0],
    [0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0],
    [0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 4, 4, 4, 8, 4, 4, 4, 8, 4, 4, 4, 8, 0, 0, 0],
    [0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 4, 4, 4, 8, 4, 4, 4, 8, 4, 4, 4, 8, 0, 0, 0],
    [0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 4, 4, 4, 8, 4, 4, 4, 8, 4, 4, 4, 8, 0, 0, 0],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 4, 4, 4, 8, 2, 2, 2, 8, 4, 4, 4, 8, 0, 0, 0],
    [0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 4, 4, 4, 8, 2, 2, 2, 8, 4, 4, 4, 8, 0, 0, 0],
    [0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 4, 4, 4, 8, 2, 2, 2, 8, 4, 4, 4, 8, 0, 0, 0],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 4, 4, 4, 8, 4, 4, 4, 8, 4, 4, 4, 8, 0, 0, 0],
    [0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 4, 4, 4, 8, 4, 4, 4, 8, 4, 4, 4, 8, 0, 0, 0],
    [0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 4, 4, 4, 8, 4, 4, 4, 8, 4, 4, 4, 8, 0, 0, 0],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0],
    [0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0],
    [0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0],
], dtype=int)

E3_OUT = np.array([
    [4, 4, 4, 8, 4, 4, 4, 8, 4, 4, 4, 8, 0, 0, 0, 8, 0, 0, 0, 8, 4, 4, 4, 8, 2, 2, 2],
    [4, 4, 4, 8, 4, 4, 4, 8, 4, 4, 4, 8, 0, 0, 0, 8, 0, 0, 0, 8, 4, 4, 4, 8, 2, 2, 2],
    [4, 4, 4, 8, 4, 4, 4, 8, 4, 4, 4, 8, 0, 0, 0, 8, 0, 0, 0, 8, 4, 4, 4, 8, 2, 2, 2],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [4, 4, 4, 8, 2, 2, 2, 8, 4, 4, 4, 8, 0, 0, 0, 8, 0, 0, 0, 8, 4, 4, 4, 8, 4, 4, 4],
    [4, 4, 4, 8, 2, 2, 2, 8, 4, 4, 4, 8, 0, 0, 0, 8, 0, 0, 0, 8, 4, 4, 4, 8, 4, 4, 4],
    [4, 4, 4, 8, 2, 2, 2, 8, 4, 4, 4, 8, 0, 0, 0, 8, 0, 0, 0, 8, 4, 4, 4, 8, 4, 4, 4],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [4, 4, 4, 8, 4, 4, 4, 8, 4, 4, 4, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0],
    [4, 4, 4, 8, 4, 4, 4, 8, 4, 4, 4, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0],
    [4, 4, 4, 8, 4, 4, 4, 8, 4, 4, 4, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 4, 4, 4, 8, 4, 4, 4, 8, 4, 4, 4, 8, 0, 0, 0],
    [0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 4, 4, 4, 8, 4, 4, 4, 8, 4, 4, 4, 8, 0, 0, 0],
    [0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 4, 4, 4, 8, 4, 4, 4, 8, 4, 4, 4, 8, 0, 0, 0],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 4, 4, 4, 8, 2, 2, 2, 8, 4, 4, 4, 8, 0, 0, 0],
    [0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 4, 4, 4, 8, 2, 2, 2, 8, 4, 4, 4, 8, 0, 0, 0],
    [0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 4, 4, 4, 8, 2, 2, 2, 8, 4, 4, 4, 8, 0, 0, 0],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 4, 4, 4, 8, 4, 4, 4, 8, 4, 4, 4, 8, 0, 0, 0],
    [0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 4, 4, 4, 8, 4, 4, 4, 8, 4, 4, 4, 8, 0, 0, 0],
    [0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 4, 4, 4, 8, 4, 4, 4, 8, 4, 4, 4, 8, 0, 0, 0],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0],
    [0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0],
    [0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0],
    [0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0],
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 6, 6, 4, 0, 0, 4, 0, 0],
    [0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 6, 6, 4, 0, 0, 4, 0, 0],
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [0, 0, 4, 0, 0, 4, 8, 8, 4, 3, 3, 4, 8, 8, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0],
    [0, 0, 4, 0, 0, 4, 8, 8, 4, 3, 3, 4, 8, 8, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0],
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [0, 0, 4, 0, 0, 4, 3, 3, 4, 6, 6, 4, 3, 3, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0],
    [0, 0, 4, 0, 0, 4, 3, 3, 4, 6, 6, 4, 3, 3, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0],
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [0, 0, 4, 0, 0, 4, 8, 8, 4, 3, 3, 4, 8, 8, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0],
    [0, 0, 4, 0, 0, 4, 8, 8, 4, 3, 3, 4, 8, 8, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0],
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0],
    [0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0],
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0],
    [0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0],
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 6, 6, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0],
    [0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 6, 6, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0],
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [6, 6, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0],
    [6, 6, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0],
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0],
    [0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0],
], dtype=int)

T_OUT = np.array([
    [0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 8, 8, 4, 3, 3, 4, 8, 8, 4, 0, 0],
    [0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 8, 8, 4, 3, 3, 4, 8, 8, 4, 0, 0],
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 3, 3, 4, 6, 6, 4, 3, 3, 4, 0, 0],
    [0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 3, 3, 4, 6, 6, 4, 3, 3, 4, 0, 0],
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [0, 0, 4, 0, 0, 4, 8, 8, 4, 3, 3, 4, 8, 8, 4, 0, 0, 4, 8, 8, 4, 3, 3, 4, 8, 8, 4, 0, 0],
    [0, 0, 4, 0, 0, 4, 8, 8, 4, 3, 3, 4, 8, 8, 4, 0, 0, 4, 8, 8, 4, 3, 3, 4, 8, 8, 4, 0, 0],
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [0, 0, 4, 0, 0, 4, 3, 3, 4, 6, 6, 4, 3, 3, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0],
    [0, 0, 4, 0, 0, 4, 3, 3, 4, 6, 6, 4, 3, 3, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0],
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [0, 0, 4, 0, 0, 4, 8, 8, 4, 3, 3, 4, 8, 8, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0],
    [0, 0, 4, 0, 0, 4, 8, 8, 4, 3, 3, 4, 8, 8, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0],
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0],
    [0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0],
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 8, 8, 4, 3, 3, 4, 8, 8, 4, 0, 0, 4, 0, 0, 4, 0, 0],
    [0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 8, 8, 4, 3, 3, 4, 8, 8, 4, 0, 0, 4, 0, 0, 4, 0, 0],
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [3, 3, 4, 8, 8, 4, 0, 0, 4, 0, 0, 4, 3, 3, 4, 6, 6, 4, 3, 3, 4, 0, 0, 4, 0, 0, 4, 0, 0],
    [3, 3, 4, 8, 8, 4, 0, 0, 4, 0, 0, 4, 3, 3, 4, 6, 6, 4, 3, 3, 4, 0, 0, 4, 0, 0, 4, 0, 0],
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [6, 6, 4, 3, 3, 4, 0, 0, 4, 0, 0, 4, 8, 8, 4, 3, 3, 4, 8, 8, 4, 0, 0, 4, 0, 0, 4, 0, 0],
    [6, 6, 4, 3, 3, 4, 0, 0, 4, 0, 0, 4, 8, 8, 4, 3, 3, 4, 8, 8, 4, 0, 0, 4, 0, 0, 4, 0, 0],
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [3, 3, 4, 8, 8, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0],
    [3, 3, 4, 8, 8, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(*args, **kwargs):
    raise NotImplementedError("Barnacles solution not available for 080")


# --- Code Golf Solution (Compressed) ---
def q(f):
    n = f.index(min(f, key=set)) + 1
    e = {u * 1j + o: f for u, f in enumerate(f[::n]) for o, f in enumerate(f[::n])}
    return [[[f or [e[(t := (u // n * 1j + o // n))], *[e[a + t - r] for r in e if (e[r] == e[a]) * 2 > abs(t - r)]][-1] for o, f in enumerate(f)] for u, f in enumerate(f)] for a in e if all((e.get(a + 1j ** u) for u, f in enumerate(f)))][-1]


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import choice, randint, sample, uniform

Integer = int

IntegerTuple = Tuple[Integer, Integer]

Grid = Tuple[Tuple[Integer]]

Cell = Tuple[Integer, IntegerTuple]

Object = FrozenSet[Cell]

Indices = FrozenSet[IntegerTuple]

Patch = Union[Object, Indices]

ContainerContainer = Container[Container]

def difference(
    a: Container,
    b: Container
) -> Container:
    """ difference """
    return type(a)(e for e in a if e not in b)

def merge(
    containers: ContainerContainer
) -> Container:
    """ merging """
    return type(containers)(e for c in containers for e in c)

def totuple(
    container: FrozenSet
) -> Tuple:
    """ conversion to tuple """
    return tuple(container)

def remove(
    value: Any,
    container: Container
) -> Container:
    """ remove item from container """
    return type(container)(e for e in container if e != value)

def interval(
    start: Integer,
    stop: Integer,
    step: Integer
) -> Tuple:
    """ range """
    return tuple(range(start, stop, step))

def apply(
    function: Callable,
    container: Container
) -> Container:
    """ apply function to each item in container """
    return type(container)(function(e) for e in container)

def mapply(
    function: Callable,
    container: ContainerContainer
) -> FrozenSet:
    """ apply and merge """
    return merge(apply(function, container))

def asindices(
    grid: Grid
) -> Indices:
    """ indices of all grid cells """
    return frozenset((i, j) for i in range(len(grid)) for j in range(len(grid[0])))

def ofcolor(
    grid: Grid,
    value: Integer
) -> Indices:
    """ indices of all grid cells with value """
    return frozenset((i, j) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value)

def toindices(
    patch: Patch
) -> Indices:
    """ indices of object cells """
    if len(patch) == 0:
        return frozenset()
    if isinstance(next(iter(patch))[1], tuple):
        return frozenset(index for value, index in patch)
    return patch

def shift(
    patch: Patch,
    directions: IntegerTuple
) -> Patch:
    """ shift patch """
    if len(patch) == 0:
        return patch
    di, dj = directions
    if isinstance(next(iter(patch))[1], tuple):
        return frozenset((value, (i + di, j + dj)) for value, (i, j) in patch)
    return frozenset((i + di, j + dj) for i, j in patch)

def dneighbors(
    loc: IntegerTuple
) -> Indices:
    """ directly adjacent indices """
    return frozenset({(loc[0] - 1, loc[1]), (loc[0] + 1, loc[1]), (loc[0], loc[1] - 1), (loc[0], loc[1] + 1)})

def ineighbors(
    loc: IntegerTuple
) -> Indices:
    """ diagonally adjacent indices """
    return frozenset({(loc[0] - 1, loc[1] - 1), (loc[0] - 1, loc[1] + 1), (loc[0] + 1, loc[1] - 1), (loc[0] + 1, loc[1] + 1)})

def neighbors(
    loc: IntegerTuple
) -> Indices:
    """ adjacent indices """
    return dneighbors(loc) | ineighbors(loc)

def fill(
    grid: Grid,
    value: Integer,
    patch: Patch
) -> Grid:
    """ fill value at indices """
    h, w = len(grid), len(grid[0])
    grid_filled = list(list(row) for row in grid)
    for i, j in toindices(patch):
        if 0 <= i < h and 0 <= j < w:
            grid_filled[i][j] = value
    return tuple(tuple(row) for row in grid_filled)

def paint(
    grid: Grid,
    obj: Object
) -> Grid:
    """ paint object to grid """
    h, w = len(grid), len(grid[0])
    grid_painted = list(list(row) for row in grid)
    for value, (i, j) in obj:
        if 0 <= i < h and 0 <= j < w:
            grid_painted[i][j] = value
    return tuple(tuple(row) for row in grid_painted)

def canvas(
    value: Integer,
    dimensions: IntegerTuple
) -> Grid:
    """ grid construction """
    return tuple(tuple(value for j in range(dimensions[1])) for i in range(dimensions[0]))

rng = []

def unifint(
    diff_lb: float,
    diff_ub: float,
    bounds: Tuple[int, int]
) -> int:
    """
    diff_lb: lower bound for difficulty, must be in range [0, diff_ub]
    diff_ub: upper bound for difficulty, must be in range [diff_lb, 1]
    bounds: interval [a, b] determining the integer values that can be sampled
    """
    a, b = bounds
    d = uniform(diff_lb, diff_ub)
    global rng
    rng.append(d)
    return min(max(a, round(a + (b - a) * d)), b)

def generate_39e1d7f9(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (5, 10))
    w = unifint(diff_lb, diff_ub, (5, 10))
    bgc, linc, dotc = sample(cols, 3)
    remcols = difference(cols, (bgc, linc, dotc))
    gi = canvas(bgc, (h, w))
    loci = randint(1, h - 2)
    locj = randint(1, w - 2)
    if h == 5:
        loci = choice((1, h - 2))
    if w == 5:
        locj = choice((1, w - 2))
    npix = unifint(diff_lb, diff_ub, (1, 8))
    ncols = unifint(diff_lb, diff_ub, (1, 7))
    ccols = sample(remcols, ncols)
    candsss = neighbors((loci, locj))
    pixs = {(loci, locj)}
    for k in range(npix):
        pixs.add(choice(totuple((mapply(dneighbors, pixs) & candsss) - pixs)))
    pixs = totuple(remove((loci, locj), pixs))
    obj = {(choice(ccols), ij) for ij in pixs}
    gi = fill(gi, dotc, {(loci, locj)})
    gi = paint(gi, obj)
    go = tuple(e for e in gi)
    noccs = unifint(diff_lb, diff_ub, (1, (h * w) // (2 * len(pixs) + 1)))
    succ = 0
    tr = 0
    maxtr = 6 * noccs
    inds = ofcolor(gi, bgc) - mapply(dneighbors, neighbors((loci, locj)))
    objn = shift(obj, (-loci, -locj))
    triedandfailed = set()
    while (tr < maxtr and succ < noccs) or succ == 0:
        lopcands = totuple(inds - triedandfailed)
        if len(lopcands) == 0:
            break
        tr += 1
        loci, locj = choice(lopcands)
        plcd = shift(objn, (loci, locj))
        plcdi = toindices(plcd)
        if plcdi.issubset(inds):
            inds = inds - (plcdi | {(loci, locj)})
            succ += 1
            gi = fill(gi, dotc, {(loci, locj)})
            go = fill(go, dotc, {(loci, locj)})
            go = paint(go, plcd)
        else:
            triedandfailed.add((loci, locj))
    hfac = unifint(diff_lb, diff_ub, (1, (30 - h + 1) // h))
    wfac = unifint(diff_lb, diff_ub, (1, (30 - w + 1) // w))
    fullh = hfac * h + h - 1
    fullw = wfac * w + w - 1
    gi2 = canvas(linc, (fullh, fullw))
    go2 = canvas(linc, (fullh, fullw))
    bd = asindices(canvas(-1, (hfac, wfac)))
    for a in range(h):
        for b in range(w):
            c = gi[a][b]
            gi2 = fill(gi2, c, shift(bd, (a * (hfac + 1), b * (wfac + 1))))
    for a in range(h):
        for b in range(w):
            c = go[a][b]
            go2 = fill(go2, c, shift(bd, (a * (hfac + 1), b * (wfac + 1))))
    gi, go = gi2, go2
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Boolean = bool

Numerical = Union[Integer, IntegerTuple]

IntegerSet = FrozenSet[Integer]

Objects = FrozenSet[Object]

Element = Union[Object, Grid]

Piece = Union[Grid, Patch]

TupleTuple = Tuple[Tuple]

ZERO = 0

ONE = 1

F = False

T = True

ORIGIN = (0, 0)

def identity(
    x: Any
) -> Any:
    """ identity function """
    return x

def multiply(
    a: Numerical,
    b: Numerical
) -> Numerical:
    """ multiplication """
    if isinstance(a, int) and isinstance(b, int):
        return a * b
    elif isinstance(a, tuple) and isinstance(b, tuple):
        return (a[0] * b[0], a[1] * b[1])
    elif isinstance(a, int) and isinstance(b, tuple):
        return (a * b[0], a * b[1])
    return (a[0] * b, a[1] * b)

def divide(
    a: Numerical,
    b: Numerical
) -> Numerical:
    """ floor division """
    if isinstance(a, int) and isinstance(b, int):
        return a // b
    elif isinstance(a, tuple) and isinstance(b, tuple):
        return (a[0] // b[0], a[1] // b[1])
    elif isinstance(a, int) and isinstance(b, tuple):
        return (a // b[0], a // b[1])
    return (a[0] // b, a[1] // b)

def invert(
    n: Numerical
) -> Numerical:
    """ inversion with respect to addition """
    return -n if isinstance(n, int) else (-n[0], -n[1])

def equality(
    a: Any,
    b: Any
) -> Boolean:
    """ equality """
    return a == b

def size(
    container: Container
) -> Integer:
    """ cardinality """
    return len(container)

def argmax(
    container: Container,
    compfunc: Callable
) -> Any:
    """ largest item by custom order """
    return max(container, key=compfunc, default=None)

def initset(
    value: Any
) -> FrozenSet:
    """ initialize container """
    return frozenset({value})

def increment(
    x: Numerical
) -> Numerical:
    """ incrementing """
    return x + 1 if isinstance(x, int) else (x[0] + 1, x[1] + 1)

def positive(
    x: Integer
) -> Boolean:
    """ positive """
    return x > 0

def sfilter(
    container: Container,
    condition: Callable
) -> Container:
    """ keep elements in container that satisfy condition """
    return type(container)(e for e in container if condition(e))

def first(
    container: Container
) -> Any:
    """ first item of container """
    return next(iter(container))

def last(
    container: Container
) -> Any:
    """ last item of container """
    return max(enumerate(container))[1]

def astuple(
    a: Integer,
    b: Integer
) -> IntegerTuple:
    """ constructs a tuple """
    return (a, b)

def pair(
    a: Tuple,
    b: Tuple
) -> TupleTuple:
    """ zipping of two tuples """
    return tuple(zip(a, b))

def branch(
    condition: Boolean,
    if_value: Any,
    else_value: Any
) -> Any:
    """ if else branching """
    return if_value if condition else else_value

def compose(
    outer: Callable,
    inner: Callable
) -> Callable:
    """ function composition """
    return lambda x: outer(inner(x))

def chain(
    h: Callable,
    g: Callable,
    f: Callable
) -> Callable:
    """ function composition with three functions """
    return lambda x: h(g(f(x)))

def matcher(
    function: Callable,
    target: Any
) -> Callable:
    """ construction of equality function """
    return lambda x: function(x) == target

def rbind(
    function: Callable,
    fixed: Any
) -> Callable:
    """ fix the rightmost argument """
    n = function.__code__.co_argcount
    if n == 2:
        return lambda x: function(x, fixed)
    elif n == 3:
        return lambda x, y: function(x, y, fixed)
    else:
        return lambda x, y, z: function(x, y, z, fixed)

def lbind(
    function: Callable,
    fixed: Any
) -> Callable:
    """ fix the leftmost argument """
    n = function.__code__.co_argcount
    if n == 2:
        return lambda y: function(fixed, y)
    elif n == 3:
        return lambda y, z: function(fixed, y, z)
    else:
        return lambda y, z, a: function(fixed, y, z, a)

def fork(
    outer: Callable,
    a: Callable,
    b: Callable
) -> Callable:
    """ creates a wrapper function """
    return lambda x: outer(a(x), b(x))

def rapply(
    functions: Container,
    value: Any
) -> Container:
    """ apply each function in container to value """
    return type(functions)(function(value) for function in functions)

def mostcolor(
    element: Element
) -> Integer:
    """ most common color """
    values = [v for r in element for v in r] if isinstance(element, tuple) else [v for v, _ in element]
    return max(set(values), key=values.count)

def leastcolor(
    element: Element
) -> Integer:
    """ least common color """
    values = [v for r in element for v in r] if isinstance(element, tuple) else [v for v, _ in element]
    return min(set(values), key=values.count)

def height(
    piece: Piece
) -> Integer:
    """ height of grid or patch """
    if len(piece) == 0:
        return 0
    if isinstance(piece, tuple):
        return len(piece)
    return lowermost(piece) - uppermost(piece) + 1

def width(
    piece: Piece
) -> Integer:
    """ width of grid or patch """
    if len(piece) == 0:
        return 0
    if isinstance(piece, tuple):
        return len(piece[0])
    return rightmost(piece) - leftmost(piece) + 1

def shape(
    piece: Piece
) -> IntegerTuple:
    """ height and width of grid or patch """
    return (height(piece), width(piece))

def colorfilter(
    objs: Objects,
    value: Integer
) -> Objects:
    """ filter objects by color """
    return frozenset(obj for obj in objs if next(iter(obj))[0] == value)

def ulcorner(
    patch: Patch
) -> IntegerTuple:
    """ index of upper left corner """
    return tuple(map(min, zip(*toindices(patch))))

def crop(
    grid: Grid,
    start: IntegerTuple,
    dims: IntegerTuple
) -> Grid:
    """ subgrid specified by start and dimension """
    return tuple(r[start[1]:start[1]+dims[1]] for r in grid[start[0]:start[0]+dims[0]])

def normalize(
    patch: Patch
) -> Patch:
    """ moves upper left corner to origin """
    if len(patch) == 0:
        return patch
    return shift(patch, (-uppermost(patch), -leftmost(patch)))

def objects(
    grid: Grid,
    univalued: Boolean,
    diagonal: Boolean,
    without_bg: Boolean
) -> Objects:
    """ objects occurring on the grid """
    bg = mostcolor(grid) if without_bg else None
    objs = set()
    occupied = set()
    h, w = len(grid), len(grid[0])
    unvisited = asindices(grid)
    diagfun = neighbors if diagonal else dneighbors
    for loc in unvisited:
        if loc in occupied:
            continue
        val = grid[loc[0]][loc[1]]
        if val == bg:
            continue
        obj = {(val, loc)}
        cands = {loc}
        while len(cands) > 0:
            neighborhood = set()
            for cand in cands:
                v = grid[cand[0]][cand[1]]
                if (val == v) if univalued else (v != bg):
                    obj.add((v, cand))
                    occupied.add(cand)
                    neighborhood |= {
                        (i, j) for i, j in diagfun(cand) if 0 <= i < h and 0 <= j < w
                    }
            cands = neighborhood - occupied
        objs.add(frozenset(obj))
    return frozenset(objs)

def uppermost(
    patch: Patch
) -> Integer:
    """ row index of uppermost occupied cell """
    return min(i for i, j in toindices(patch))

def lowermost(
    patch: Patch
) -> Integer:
    """ row index of lowermost occupied cell """
    return max(i for i, j in toindices(patch))

def leftmost(
    patch: Patch
) -> Integer:
    """ column index of leftmost occupied cell """
    return min(j for i, j in toindices(patch))

def rightmost(
    patch: Patch
) -> Integer:
    """ column index of rightmost occupied cell """
    return max(j for i, j in toindices(patch))

def vline(
    patch: Patch
) -> Boolean:
    """ whether the piece forms a vertical line """
    return height(patch) == len(patch) and width(patch) == 1

def hline(
    patch: Patch
) -> Boolean:
    """ whether the piece forms a horizontal line """
    return width(patch) == len(patch) and height(patch) == 1

def palette(
    element: Element
) -> IntegerSet:
    """ colors occurring in object or grid """
    if isinstance(element, tuple):
        return frozenset({v for r in element for v in r})
    return frozenset({v for v, _ in element})

def numcolors(
    element: Element
) -> IntegerSet:
    """ number of colors occurring in object or grid """
    return len(palette(element))

def color(
    obj: Object
) -> Integer:
    """ color of object """
    return next(iter(obj))[0]

def dmirror(
    piece: Piece
) -> Piece:
    """ mirroring along diagonal """
    if isinstance(piece, tuple):
        return tuple(zip(*piece))
    a, b = ulcorner(piece)
    if isinstance(next(iter(piece))[1], tuple):
        return frozenset((v, (j - b + a, i - a + b)) for v, (i, j) in piece)
    return frozenset((j - b + a, i - a + b) for i, j in piece)

def hupscale(
    grid: Grid,
    factor: Integer
) -> Grid:
    """ upscale grid horizontally """
    upscaled_grid = tuple()
    for row in grid:
        upscaled_row = tuple()
        for value in row:
            upscaled_row = upscaled_row + tuple(value for num in range(factor))
        upscaled_grid = upscaled_grid + (upscaled_row,)
    return upscaled_grid

def vupscale(
    grid: Grid,
    factor: Integer
) -> Grid:
    """ upscale grid vertically """
    upscaled_grid = tuple()
    for row in grid:
        upscaled_grid = upscaled_grid + tuple(row for num in range(factor))
    return upscaled_grid

def frontiers(
    grid: Grid
) -> Objects:
    """ set of frontiers """
    h, w = len(grid), len(grid[0])
    row_indices = tuple(i for i, r in enumerate(grid) if len(set(r)) == 1)
    column_indices = tuple(j for j, c in enumerate(dmirror(grid)) if len(set(c)) == 1)
    hfrontiers = frozenset({frozenset({(grid[i][j], (i, j)) for j in range(w)}) for i in row_indices})
    vfrontiers = frozenset({frozenset({(grid[i][j], (i, j)) for i in range(h)}) for j in column_indices})
    return hfrontiers | vfrontiers

def compress(
    grid: Grid
) -> Grid:
    """ removes frontiers from grid """
    ri = tuple(i for i, r in enumerate(grid) if len(set(r)) == 1)
    ci = tuple(j for j, c in enumerate(dmirror(grid)) if len(set(c)) == 1)
    return tuple(tuple(v for j, v in enumerate(r) if j not in ci) for i, r in enumerate(grid) if i not in ri)

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_39e1d7f9(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = compress(I)
    x1 = objects(x0, F, F, T)
    x2 = argmax(x1, numcolors)
    x3 = remove(x2, x1)
    x4 = merge(x3)
    x5 = size(x4)
    x6 = positive(x5)
    x7 = astuple(color, x4)
    x8 = astuple(leastcolor, x2)
    x9 = branch(x6, x7, x8)
    x10 = compose(initset, first)
    x11 = fork(rapply, x10, last)
    x12 = compose(first, x11)
    x13 = x12(x9)
    x14 = normalize(x2)
    x15 = matcher(first, x13)
    x16 = sfilter(x14, x15)
    x17 = ulcorner(x16)
    x18 = invert(x17)
    x19 = shift(x14, x18)
    x20 = lbind(shift, x19)
    x21 = objects(x0, T, F, T)
    x22 = colorfilter(x21, x13)
    x23 = apply(ulcorner, x22)
    x24 = mapply(x20, x23)
    x25 = paint(x0, x24)
    x26 = height(x0)
    x27 = frontiers(I)
    x28 = sfilter(x27, hline)
    x29 = size(x28)
    x30 = increment(x29)
    x31 = divide(x26, x30)
    x32 = width(x0)
    x33 = frontiers(I)
    x34 = sfilter(x33, vline)
    x35 = size(x34)
    x36 = increment(x35)
    x37 = divide(x32, x36)
    x38 = rbind(multiply, x37)
    x39 = rbind(divide, x37)
    x40 = compose(x38, x39)
    x41 = fork(equality, x40, identity)
    x42 = compose(x41, first)
    x43 = rbind(multiply, x31)
    x44 = rbind(divide, x31)
    x45 = compose(x43, x44)
    x46 = fork(equality, x45, identity)
    x47 = compose(x46, first)
    x48 = lbind(interval, ZERO)
    x49 = rbind(x48, ONE)
    x50 = compose(x49, size)
    x51 = fork(pair, x50, identity)
    x52 = lbind(apply, last)
    x53 = rbind(sfilter, x42)
    x54 = chain(x52, x53, x51)
    x55 = compose(x54, last)
    x56 = height(x25)
    x57 = interval(ZERO, x56, ONE)
    x58 = pair(x57, x25)
    x59 = sfilter(x58, x47)
    x60 = apply(x55, x59)
    x61 = increment(x37)
    x62 = hupscale(x60, x61)
    x63 = increment(x31)
    x64 = vupscale(x62, x63)
    x65 = frontiers(I)
    x66 = merge(x65)
    x67 = paint(x64, x66)
    x68 = shape(I)
    x69 = crop(x67, ORIGIN, x68)
    return x69


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_39e1d7f9(inp)
        assert pred == _to_grid(expected), f"{name} failed"
