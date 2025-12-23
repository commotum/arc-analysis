# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "c1d99e64"
SERIAL = "303"
URL    = "https://arcprize.org/play?task=c1d99e64"

# --- Code Golf Concepts ---
CONCEPTS = [
    "draw_line_from_border",
    "detect_grid",
]

# --- Example Grids ---
E1_IN = np.array([
    [1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1],
    [1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1],
    [1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0],
    [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
    [1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
], dtype=int)

E1_OUT = np.array([
    [1, 0, 0, 0, 1, 1, 1, 1, 2, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1],
    [1, 0, 1, 0, 1, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1],
    [1, 1, 1, 1, 0, 0, 1, 1, 2, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0],
    [1, 0, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 1, 1, 0, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 0, 1, 0, 1, 1, 0, 2, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 1, 0, 2, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
    [1, 1, 0, 0, 1, 1, 1, 1, 2, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [1, 1, 1, 0, 0, 1, 1, 1, 2, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 0, 0, 1, 1, 0, 0, 2, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 0, 1, 2, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
], dtype=int)

E2_IN = np.array([
    [8, 8, 8, 8, 0, 8, 8, 8, 8, 8, 0, 0, 8, 8],
    [0, 8, 0, 0, 0, 0, 8, 8, 8, 8, 0, 8, 8, 8],
    [8, 8, 0, 8, 0, 8, 8, 8, 8, 8, 0, 0, 8, 8],
    [8, 0, 8, 8, 0, 8, 8, 0, 0, 8, 0, 8, 8, 0],
    [8, 8, 8, 8, 0, 8, 8, 0, 0, 0, 0, 8, 8, 8],
    [8, 8, 8, 0, 0, 8, 8, 0, 8, 0, 0, 8, 8, 8],
    [8, 0, 8, 8, 0, 8, 8, 8, 8, 8, 0, 0, 0, 8],
    [8, 8, 0, 0, 0, 8, 0, 0, 8, 8, 0, 0, 8, 8],
    [8, 0, 0, 8, 0, 8, 8, 8, 0, 8, 0, 8, 8, 8],
    [8, 8, 0, 8, 0, 8, 8, 8, 8, 8, 0, 0, 8, 0],
    [0, 8, 0, 8, 0, 0, 0, 0, 0, 0, 0, 8, 0, 8],
    [8, 8, 8, 8, 0, 8, 8, 8, 8, 8, 0, 0, 8, 0],
], dtype=int)

E2_OUT = np.array([
    [8, 8, 8, 8, 2, 8, 8, 8, 8, 8, 2, 0, 8, 8],
    [0, 8, 0, 0, 2, 0, 8, 8, 8, 8, 2, 8, 8, 8],
    [8, 8, 0, 8, 2, 8, 8, 8, 8, 8, 2, 0, 8, 8],
    [8, 0, 8, 8, 2, 8, 8, 0, 0, 8, 2, 8, 8, 0],
    [8, 8, 8, 8, 2, 8, 8, 0, 0, 0, 2, 8, 8, 8],
    [8, 8, 8, 0, 2, 8, 8, 0, 8, 0, 2, 8, 8, 8],
    [8, 0, 8, 8, 2, 8, 8, 8, 8, 8, 2, 0, 0, 8],
    [8, 8, 0, 0, 2, 8, 0, 0, 8, 8, 2, 0, 8, 8],
    [8, 0, 0, 8, 2, 8, 8, 8, 0, 8, 2, 8, 8, 8],
    [8, 8, 0, 8, 2, 8, 8, 8, 8, 8, 2, 0, 8, 0],
    [0, 8, 0, 8, 2, 0, 0, 0, 0, 0, 2, 8, 0, 8],
    [8, 8, 8, 8, 2, 8, 8, 8, 8, 8, 2, 0, 8, 0],
], dtype=int)

E3_IN = np.array([
    [3, 0, 3, 3, 3, 3, 3, 0, 3, 3, 3, 0, 3, 0, 3],
    [3, 0, 3, 0, 3, 3, 3, 0, 3, 0, 3, 0, 0, 3, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 0, 0, 3, 0, 0, 3, 3, 0, 3, 0, 3, 3, 0, 0],
    [3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3, 3],
    [3, 0, 3, 3, 3, 3, 3, 3, 0, 0, 3, 3, 0, 3, 3],
    [0, 0, 3, 0, 3, 0, 3, 0, 3, 0, 0, 3, 3, 3, 0],
    [3, 0, 0, 3, 3, 3, 0, 0, 3, 0, 3, 3, 0, 0, 3],
    [3, 0, 3, 3, 3, 3, 3, 0, 3, 3, 3, 3, 3, 0, 3],
    [3, 0, 0, 3, 3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 0],
    [3, 0, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3, 0, 3, 3],
    [3, 0, 3, 3, 3, 0, 3, 0, 0, 3, 0, 3, 3, 3, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 0, 3, 0, 0, 3, 0, 3, 3, 0, 3, 3, 3, 3, 0],
    [3, 0, 0, 3, 0, 3, 3, 0, 3, 0, 3, 3, 0, 0, 3],
    [3, 0, 0, 3, 3, 3, 3, 3, 0, 3, 3, 0, 0, 3, 3],
    [0, 0, 3, 3, 0, 3, 3, 0, 0, 3, 0, 3, 0, 3, 0],
], dtype=int)

E3_OUT = np.array([
    [3, 2, 3, 3, 3, 3, 3, 0, 3, 3, 3, 0, 3, 0, 3],
    [3, 2, 3, 0, 3, 3, 3, 0, 3, 0, 3, 0, 0, 3, 0],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [3, 2, 0, 3, 0, 0, 3, 3, 0, 3, 0, 3, 3, 0, 0],
    [3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3, 3],
    [3, 2, 3, 3, 3, 3, 3, 3, 0, 0, 3, 3, 0, 3, 3],
    [0, 2, 3, 0, 3, 0, 3, 0, 3, 0, 0, 3, 3, 3, 0],
    [3, 2, 0, 3, 3, 3, 0, 0, 3, 0, 3, 3, 0, 0, 3],
    [3, 2, 3, 3, 3, 3, 3, 0, 3, 3, 3, 3, 3, 0, 3],
    [3, 2, 0, 3, 3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 0],
    [3, 2, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3, 0, 3, 3],
    [3, 2, 3, 3, 3, 0, 3, 0, 0, 3, 0, 3, 3, 3, 0],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [3, 2, 3, 0, 0, 3, 0, 3, 3, 0, 3, 3, 3, 3, 0],
    [3, 2, 0, 3, 0, 3, 3, 0, 3, 0, 3, 3, 0, 0, 3],
    [3, 2, 0, 3, 3, 3, 3, 3, 0, 3, 3, 0, 0, 3, 3],
    [0, 2, 3, 3, 0, 3, 3, 0, 0, 3, 0, 3, 0, 3, 0],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [4, 0, 4, 0, 4, 4, 0, 0, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 0, 4, 4, 0, 4, 0, 0],
    [4, 4, 4, 0, 0, 4, 0, 4, 4, 0, 4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 0, 4, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 0, 4, 4, 4, 0, 0, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 4, 4, 0, 4, 4, 0],
    [4, 4, 0, 4, 4, 4, 0, 0, 0, 0, 4, 4, 4, 4, 0, 4, 4, 4, 0, 4, 4, 0, 4, 4, 4],
    [4, 4, 4, 0, 4, 4, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 4, 0, 4, 0, 4, 0, 4],
    [4, 0, 0, 4, 0, 4, 0, 4, 4, 4, 4, 4, 4, 0, 4, 0, 4, 4, 4, 0, 4, 0, 4, 4, 4],
    [4, 4, 4, 4, 4, 0, 0, 4, 0, 4, 0, 0, 4, 4, 0, 0, 4, 4, 4, 0, 0, 0, 0, 4, 0],
    [0, 4, 4, 0, 4, 4, 0, 4, 4, 0, 4, 4, 0, 4, 4, 0, 0, 4, 0, 4, 0, 0, 4, 0, 4],
    [4, 4, 4, 0, 4, 4, 0, 0, 4, 4, 4, 4, 4, 0, 0, 4, 0, 4, 4, 4, 0, 0, 4, 4, 4],
    [4, 0, 4, 4, 4, 0, 0, 4, 0, 4, 4, 0, 4, 4, 0, 4, 4, 0, 4, 4, 0, 0, 0, 0, 4],
    [4, 4, 0, 4, 0, 0, 0, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 0, 4, 4, 4],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 4, 4, 0, 0, 0, 0, 0, 4, 4, 4, 4, 0, 4, 4, 0, 0, 4, 4, 4, 4, 0, 0, 4, 4],
    [4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 0, 4, 0, 4, 4, 0, 4, 4, 4, 4, 0, 4, 4, 4],
    [4, 4, 4, 4, 4, 0, 0, 4, 0, 4, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 4, 0, 4],
    [0, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 0, 4, 0, 4, 4, 0, 4, 4, 4, 0, 4, 4, 0],
    [0, 4, 4, 4, 4, 0, 0, 4, 4, 4, 0, 4, 0, 4, 0, 4, 4, 4, 4, 4, 4, 0, 0, 4, 4],
    [4, 4, 4, 0, 4, 4, 0, 0, 4, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0],
    [4, 4, 0, 4, 4, 4, 0, 4, 4, 0, 4, 4, 4, 0, 4, 4, 4, 0, 4, 4, 0, 0, 0, 4, 4],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 4, 4, 4, 0, 4, 0, 4, 0, 4, 4, 4, 0, 0, 0, 0, 4, 0, 4, 4, 4, 0, 4, 4, 4],
    [0, 4, 4, 4, 4, 4, 0, 4, 0, 4, 0, 4, 4, 0, 4, 4, 0, 4, 4, 0, 4, 0, 4, 4, 4],
    [4, 4, 4, 4, 4, 4, 0, 4, 4, 0, 0, 0, 0, 4, 4, 4, 0, 0, 4, 4, 4, 0, 4, 4, 0],
    [4, 0, 4, 0, 4, 4, 0, 4, 0, 0, 0, 4, 4, 4, 4, 4, 0, 4, 0, 4, 4, 0, 0, 4, 0],
    [4, 4, 0, 4, 0, 4, 0, 0, 4, 0, 4, 4, 0, 4, 4, 0, 0, 0, 4, 0, 4, 0, 4, 4, 4],
    [4, 0, 0, 4, 4, 4, 0, 4, 0, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 0, 0, 0, 4, 4, 4],
], dtype=int)

T_OUT = np.array([
    [4, 0, 4, 0, 4, 4, 2, 0, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 0, 4, 4, 2, 4, 0, 0],
    [4, 4, 4, 0, 0, 4, 2, 4, 4, 0, 4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 2, 4, 0, 0],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [4, 0, 4, 4, 4, 0, 2, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 4, 4, 2, 4, 4, 0],
    [4, 4, 0, 4, 4, 4, 2, 0, 0, 0, 4, 4, 4, 4, 0, 4, 4, 4, 0, 4, 4, 2, 4, 4, 4],
    [4, 4, 4, 0, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 4, 0, 4, 2, 4, 0, 4],
    [4, 0, 0, 4, 0, 4, 2, 4, 4, 4, 4, 4, 4, 0, 4, 0, 4, 4, 4, 0, 4, 2, 4, 4, 4],
    [4, 4, 4, 4, 4, 0, 2, 4, 0, 4, 0, 0, 4, 4, 0, 0, 4, 4, 4, 0, 0, 2, 0, 4, 0],
    [0, 4, 4, 0, 4, 4, 2, 4, 4, 0, 4, 4, 0, 4, 4, 0, 0, 4, 0, 4, 0, 2, 4, 0, 4],
    [4, 4, 4, 0, 4, 4, 2, 0, 4, 4, 4, 4, 4, 0, 0, 4, 0, 4, 4, 4, 0, 2, 4, 4, 4],
    [4, 0, 4, 4, 4, 0, 2, 4, 0, 4, 4, 0, 4, 4, 0, 4, 4, 0, 4, 4, 0, 2, 0, 0, 4],
    [4, 4, 0, 4, 0, 0, 2, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 2, 4, 4, 4],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [0, 4, 4, 0, 0, 0, 2, 0, 4, 4, 4, 4, 0, 4, 4, 0, 0, 4, 4, 4, 4, 2, 0, 4, 4],
    [4, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 0, 4, 0, 4, 4, 0, 4, 4, 4, 4, 2, 4, 4, 4],
    [4, 4, 4, 4, 4, 0, 2, 4, 0, 4, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 0, 2, 4, 0, 4],
    [0, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 4, 0, 4, 0, 4, 4, 0, 4, 4, 4, 2, 4, 4, 0],
    [0, 4, 4, 4, 4, 0, 2, 4, 4, 4, 0, 4, 0, 4, 0, 4, 4, 4, 4, 4, 4, 2, 0, 4, 4],
    [4, 4, 4, 0, 4, 4, 2, 0, 4, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 0, 0, 0],
    [4, 4, 0, 4, 4, 4, 2, 4, 4, 0, 4, 4, 4, 0, 4, 4, 4, 0, 4, 4, 0, 2, 0, 4, 4],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [4, 4, 4, 4, 0, 4, 2, 4, 0, 4, 4, 4, 0, 0, 0, 0, 4, 0, 4, 4, 4, 2, 4, 4, 4],
    [0, 4, 4, 4, 4, 4, 2, 4, 0, 4, 0, 4, 4, 0, 4, 4, 0, 4, 4, 0, 4, 2, 4, 4, 4],
    [4, 4, 4, 4, 4, 4, 2, 4, 4, 0, 0, 0, 0, 4, 4, 4, 0, 0, 4, 4, 4, 2, 4, 4, 0],
    [4, 0, 4, 0, 4, 4, 2, 4, 0, 0, 0, 4, 4, 4, 4, 4, 0, 4, 0, 4, 4, 2, 0, 4, 0],
    [4, 4, 0, 4, 0, 4, 2, 0, 4, 0, 4, 4, 0, 4, 4, 0, 0, 0, 4, 0, 4, 2, 4, 4, 4],
    [4, 0, 0, 4, 4, 4, 2, 4, 0, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 0, 0, 2, 4, 4, 4],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(j,A=range):
 c,E=len(j),len(j[0])
 for k in A(c):
  if sum(j[k])==0:j[k]=[2]*E
 for W in A(E):
  if all(j[k][W]in[0,2]for k in A(c)):
   for k in A(c):j[k][W]=2
 return j


# --- Code Golf Solution (Compressed) ---
def q(g):
    return [[[2, e][c > [0] * 99 < l] for *c, e in zip(*g, l)] for l in g]


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import choice, sample, uniform

Integer = int

IntegerTuple = Tuple[Integer, Integer]

Grid = Tuple[Tuple[Integer]]

Cell = Tuple[Integer, IntegerTuple]

Object = FrozenSet[Cell]

Objects = FrozenSet[Object]

Indices = FrozenSet[IntegerTuple]

Patch = Union[Object, Indices]

Piece = Union[Grid, Patch]

ContainerContainer = Container[Container]

def combine(
    a: Container,
    b: Container
) -> Container:
    """ union """
    return type(a)((*a, *b))

def size(
    container: Container
) -> Integer:
    """ cardinality """
    return len(container)

def merge(
    containers: ContainerContainer
) -> Container:
    """ merging """
    return type(containers)(e for c in containers for e in c)

def toivec(
    i: Integer
) -> IntegerTuple:
    """ vector pointing vertically """
    return (i, 0)

def tojvec(
    j: Integer
) -> IntegerTuple:
    """ vector pointing horizontally """
    return (0, j)

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

def compose(
    outer: Callable,
    inner: Callable
) -> Callable:
    """ function composition """
    return lambda x: outer(inner(x))

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

def colorfilter(
    objs: Objects,
    value: Integer
) -> Objects:
    """ filter objects by color """
    return frozenset(obj for obj in objs if next(iter(obj))[0] == value)

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

def ulcorner(
    patch: Patch
) -> IntegerTuple:
    """ index of upper left corner """
    return tuple(map(min, zip(*toindices(patch))))

def toindices(
    patch: Patch
) -> Indices:
    """ indices of object cells """
    if len(patch) == 0:
        return frozenset()
    if isinstance(next(iter(patch))[1], tuple):
        return frozenset(index for value, index in patch)
    return patch

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

def canvas(
    value: Integer,
    dimensions: IntegerTuple
) -> Grid:
    """ grid construction """
    return tuple(tuple(value for j in range(dimensions[1])) for i in range(dimensions[0]))

def vfrontier(
    location: IntegerTuple
) -> Indices:
    """ vertical frontier """
    return frozenset((i, location[1]) for i in range(30))

def hfrontier(
    location: IntegerTuple
) -> Indices:
    """ horizontal frontier """
    return frozenset((location[0], j) for j in range(30))

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

def generate_c1d99e64(diff_lb: float, diff_ub: float) -> dict:
    dim_bounds = (4, 30)
    colopts = remove(2, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, dim_bounds)
    w = unifint(diff_lb, diff_ub, dim_bounds)
    nofrontcol = choice(colopts)
    noisefrontcol = choice(remove(nofrontcol, colopts))
    gi = canvas(nofrontcol, (h, w))
    cands = totuple(asindices(gi))
    horifront_bounds = (1, h//4)
    vertifront_bounds = (1, w//4)
    nhf = unifint(diff_lb, diff_ub, horifront_bounds)
    nvf = unifint(diff_lb, diff_ub, vertifront_bounds)
    vfs = mapply(compose(vfrontier, tojvec), sample(interval(0, w, 1), nvf))
    hfs = mapply(compose(hfrontier, toivec), sample(interval(0, h, 1), nhf))
    gi = fill(gi, noisefrontcol, combine(vfs, hfs))
    cands = totuple(ofcolor(gi, nofrontcol))
    kk = size(cands)
    midp = (h * w) // 2
    noise_bounds = (0, max(0, kk - midp - 1))
    num_noise = unifint(diff_lb, diff_ub, noise_bounds)
    noise = sample(cands, num_noise)
    gi = fill(gi, noisefrontcol, noise)
    go = fill(gi, 2, merge(colorfilter(frontiers(gi), noisefrontcol)))
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
TWO = 2

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_c1d99e64(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = frontiers(I)
    x1 = merge(x0)
    x2 = fill(I, TWO, x1)
    return x2


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_c1d99e64(inp)
        assert pred == _to_grid(expected), f"{name} failed"
