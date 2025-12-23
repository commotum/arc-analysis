# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "b8825c91"
SERIAL = "287"
URL    = "https://arcprize.org/play?task=b8825c91"

# --- Code Golf Concepts ---
CONCEPTS = [
    "pattern_completion",
    "pattern_rotation",
    "pattern_reflection",
]

# --- Example Grids ---
E1_IN = np.array([
    [9, 9, 6, 5, 9, 6, 7, 7, 7, 7, 6, 9, 5, 6, 9, 9],
    [9, 1, 5, 5, 6, 1, 7, 9, 9, 7, 1, 6, 5, 5, 1, 9],
    [6, 5, 1, 9, 7, 7, 3, 3, 3, 3, 7, 7, 9, 1, 5, 6],
    [5, 5, 9, 3, 7, 9, 3, 3, 3, 3, 9, 7, 3, 9, 5, 5],
    [9, 6, 7, 7, 3, 8, 9, 1, 1, 9, 8, 3, 7, 7, 6, 9],
    [6, 1, 7, 9, 8, 3, 1, 1, 1, 1, 4, 4, 4, 4, 1, 6],
    [7, 7, 3, 3, 9, 1, 6, 6, 6, 6, 4, 4, 4, 4, 7, 7],
    [7, 9, 3, 3, 1, 1, 6, 1, 1, 6, 4, 4, 4, 4, 9, 7],
    [7, 9, 3, 3, 1, 1, 6, 1, 1, 6, 1, 1, 3, 3, 9, 7],
    [7, 7, 3, 3, 9, 1, 6, 6, 6, 6, 1, 9, 3, 3, 7, 7],
    [6, 1, 7, 9, 8, 3, 1, 1, 1, 1, 4, 4, 4, 7, 1, 6],
    [9, 6, 7, 7, 3, 8, 9, 1, 1, 9, 4, 4, 4, 7, 6, 9],
    [5, 5, 9, 3, 7, 9, 3, 3, 3, 3, 4, 4, 4, 9, 5, 5],
    [6, 5, 1, 9, 7, 7, 3, 3, 3, 3, 4, 4, 4, 1, 5, 6],
    [9, 1, 5, 5, 6, 1, 7, 9, 9, 7, 1, 6, 5, 5, 1, 9],
    [9, 9, 6, 5, 9, 6, 7, 7, 7, 7, 6, 9, 5, 6, 9, 9],
], dtype=int)

E1_OUT = np.array([
    [9, 9, 6, 5, 9, 6, 7, 7, 7, 7, 6, 9, 5, 6, 9, 9],
    [9, 1, 5, 5, 6, 1, 7, 9, 9, 7, 1, 6, 5, 5, 1, 9],
    [6, 5, 1, 9, 7, 7, 3, 3, 3, 3, 7, 7, 9, 1, 5, 6],
    [5, 5, 9, 3, 7, 9, 3, 3, 3, 3, 9, 7, 3, 9, 5, 5],
    [9, 6, 7, 7, 3, 8, 9, 1, 1, 9, 8, 3, 7, 7, 6, 9],
    [6, 1, 7, 9, 8, 3, 1, 1, 1, 1, 3, 8, 9, 7, 1, 6],
    [7, 7, 3, 3, 9, 1, 6, 6, 6, 6, 1, 9, 3, 3, 7, 7],
    [7, 9, 3, 3, 1, 1, 6, 1, 1, 6, 1, 1, 3, 3, 9, 7],
    [7, 9, 3, 3, 1, 1, 6, 1, 1, 6, 1, 1, 3, 3, 9, 7],
    [7, 7, 3, 3, 9, 1, 6, 6, 6, 6, 1, 9, 3, 3, 7, 7],
    [6, 1, 7, 9, 8, 3, 1, 1, 1, 1, 3, 8, 9, 7, 1, 6],
    [9, 6, 7, 7, 3, 8, 9, 1, 1, 9, 8, 3, 7, 7, 6, 9],
    [5, 5, 9, 3, 7, 9, 3, 3, 3, 3, 9, 7, 3, 9, 5, 5],
    [6, 5, 1, 9, 7, 7, 3, 3, 3, 3, 7, 7, 9, 1, 5, 6],
    [9, 1, 5, 5, 6, 1, 7, 9, 9, 7, 1, 6, 5, 5, 1, 9],
    [9, 9, 6, 5, 9, 6, 7, 7, 7, 7, 6, 9, 5, 6, 9, 9],
], dtype=int)

E2_IN = np.array([
    [9, 9, 6, 1, 8, 9, 6, 6, 6, 6, 9, 8, 1, 6, 9, 9],
    [9, 6, 1, 3, 9, 6, 6, 1, 1, 6, 6, 9, 3, 1, 6, 9],
    [6, 4, 4, 2, 6, 6, 8, 8, 8, 8, 6, 6, 2, 5, 1, 6],
    [1, 4, 4, 8, 6, 1, 8, 2, 2, 8, 1, 6, 8, 2, 3, 1],
    [8, 4, 4, 6, 7, 1, 5, 5, 5, 5, 1, 7, 6, 6, 9, 8],
    [9, 6, 6, 1, 1, 1, 5, 5, 5, 5, 1, 1, 1, 6, 6, 9],
    [6, 6, 8, 8, 5, 5, 9, 5, 5, 9, 5, 5, 8, 8, 6, 6],
    [6, 1, 8, 2, 5, 5, 5, 8, 8, 5, 5, 5, 2, 8, 1, 6],
    [6, 1, 8, 2, 5, 5, 5, 8, 8, 5, 5, 4, 4, 4, 1, 6],
    [6, 6, 8, 8, 5, 5, 9, 5, 5, 9, 5, 4, 4, 4, 6, 6],
    [9, 6, 6, 1, 1, 1, 5, 5, 5, 5, 1, 1, 1, 6, 6, 9],
    [8, 9, 6, 6, 7, 1, 5, 5, 5, 5, 1, 7, 6, 6, 9, 8],
    [1, 3, 2, 8, 6, 1, 8, 2, 2, 8, 1, 6, 8, 2, 3, 1],
    [6, 1, 5, 2, 6, 6, 8, 8, 8, 8, 6, 6, 2, 5, 1, 6],
    [9, 6, 1, 3, 9, 6, 6, 1, 1, 6, 6, 9, 3, 1, 6, 9],
    [9, 9, 6, 1, 8, 9, 6, 6, 6, 6, 9, 8, 1, 6, 9, 9],
], dtype=int)

E2_OUT = np.array([
    [9, 9, 6, 1, 8, 9, 6, 6, 6, 6, 9, 8, 1, 6, 9, 9],
    [9, 6, 1, 3, 9, 6, 6, 1, 1, 6, 6, 9, 3, 1, 6, 9],
    [6, 1, 5, 2, 6, 6, 8, 8, 8, 8, 6, 6, 2, 5, 1, 6],
    [1, 3, 2, 8, 6, 1, 8, 2, 2, 8, 1, 6, 8, 2, 3, 1],
    [8, 9, 6, 6, 7, 1, 5, 5, 5, 5, 1, 7, 6, 6, 9, 8],
    [9, 6, 6, 1, 1, 1, 5, 5, 5, 5, 1, 1, 1, 6, 6, 9],
    [6, 6, 8, 8, 5, 5, 9, 5, 5, 9, 5, 5, 8, 8, 6, 6],
    [6, 1, 8, 2, 5, 5, 5, 8, 8, 5, 5, 5, 2, 8, 1, 6],
    [6, 1, 8, 2, 5, 5, 5, 8, 8, 5, 5, 5, 2, 8, 1, 6],
    [6, 6, 8, 8, 5, 5, 9, 5, 5, 9, 5, 5, 8, 8, 6, 6],
    [9, 6, 6, 1, 1, 1, 5, 5, 5, 5, 1, 1, 1, 6, 6, 9],
    [8, 9, 6, 6, 7, 1, 5, 5, 5, 5, 1, 7, 6, 6, 9, 8],
    [1, 3, 2, 8, 6, 1, 8, 2, 2, 8, 1, 6, 8, 2, 3, 1],
    [6, 1, 5, 2, 6, 6, 8, 8, 8, 8, 6, 6, 2, 5, 1, 6],
    [9, 6, 1, 3, 9, 6, 6, 1, 1, 6, 6, 9, 3, 1, 6, 9],
    [9, 9, 6, 1, 8, 9, 6, 6, 6, 6, 9, 8, 1, 6, 9, 9],
], dtype=int)

E3_IN = np.array([
    [9, 3, 9, 9, 2, 8, 7, 8, 8, 7, 8, 2, 9, 9, 3, 9],
    [3, 9, 9, 3, 8, 8, 8, 5, 5, 8, 8, 8, 3, 9, 9, 3],
    [9, 9, 2, 8, 7, 8, 2, 2, 2, 2, 8, 7, 8, 2, 9, 9],
    [9, 3, 8, 8, 8, 5, 2, 1, 1, 2, 5, 8, 8, 8, 3, 9],
    [2, 8, 7, 8, 2, 5, 9, 7, 7, 9, 5, 2, 8, 7, 8, 2],
    [8, 8, 8, 5, 5, 5, 7, 6, 6, 7, 5, 5, 5, 8, 8, 8],
    [7, 8, 2, 2, 9, 7, 1, 1, 1, 1, 7, 9, 4, 4, 8, 7],
    [8, 5, 2, 1, 7, 6, 1, 3, 3, 1, 6, 7, 4, 4, 5, 8],
    [8, 5, 2, 1, 7, 6, 1, 3, 3, 1, 6, 7, 4, 4, 5, 8],
    [7, 8, 2, 2, 9, 7, 1, 1, 1, 1, 7, 9, 4, 4, 8, 7],
    [8, 8, 8, 5, 5, 5, 7, 6, 6, 7, 5, 5, 5, 8, 8, 8],
    [2, 8, 4, 4, 4, 4, 9, 7, 7, 9, 5, 2, 8, 7, 8, 2],
    [9, 3, 4, 4, 4, 4, 2, 1, 1, 2, 5, 8, 8, 8, 3, 9],
    [9, 9, 4, 4, 4, 4, 2, 2, 2, 2, 8, 7, 8, 2, 9, 9],
    [3, 9, 4, 4, 4, 4, 8, 5, 5, 8, 8, 8, 3, 9, 9, 3],
    [9, 3, 9, 9, 2, 8, 7, 8, 8, 7, 8, 2, 9, 9, 3, 9],
], dtype=int)

E3_OUT = np.array([
    [9, 3, 9, 9, 2, 8, 7, 8, 8, 7, 8, 2, 9, 9, 3, 9],
    [3, 9, 9, 3, 8, 8, 8, 5, 5, 8, 8, 8, 3, 9, 9, 3],
    [9, 9, 2, 8, 7, 8, 2, 2, 2, 2, 8, 7, 8, 2, 9, 9],
    [9, 3, 8, 8, 8, 5, 2, 1, 1, 2, 5, 8, 8, 8, 3, 9],
    [2, 8, 7, 8, 2, 5, 9, 7, 7, 9, 5, 2, 8, 7, 8, 2],
    [8, 8, 8, 5, 5, 5, 7, 6, 6, 7, 5, 5, 5, 8, 8, 8],
    [7, 8, 2, 2, 9, 7, 1, 1, 1, 1, 7, 9, 2, 2, 8, 7],
    [8, 5, 2, 1, 7, 6, 1, 3, 3, 1, 6, 7, 1, 2, 5, 8],
    [8, 5, 2, 1, 7, 6, 1, 3, 3, 1, 6, 7, 1, 2, 5, 8],
    [7, 8, 2, 2, 9, 7, 1, 1, 1, 1, 7, 9, 2, 2, 8, 7],
    [8, 8, 8, 5, 5, 5, 7, 6, 6, 7, 5, 5, 5, 8, 8, 8],
    [2, 8, 7, 8, 2, 5, 9, 7, 7, 9, 5, 2, 8, 7, 8, 2],
    [9, 3, 8, 8, 8, 5, 2, 1, 1, 2, 5, 8, 8, 8, 3, 9],
    [9, 9, 2, 8, 7, 8, 2, 2, 2, 2, 8, 7, 8, 2, 9, 9],
    [3, 9, 9, 3, 8, 8, 8, 5, 5, 8, 8, 8, 3, 9, 9, 3],
    [9, 3, 9, 9, 2, 8, 7, 8, 8, 7, 8, 2, 9, 9, 3, 9],
], dtype=int)

E4_IN = np.array([
    [2, 2, 7, 6, 8, 9, 9, 1, 1, 9, 9, 8, 6, 7, 2, 2],
    [2, 1, 6, 2, 9, 5, 1, 1, 1, 1, 4, 4, 4, 4, 1, 2],
    [7, 6, 3, 3, 9, 1, 6, 6, 6, 6, 4, 4, 4, 4, 6, 7],
    [6, 2, 3, 8, 1, 1, 6, 6, 6, 6, 4, 4, 4, 4, 2, 6],
    [8, 9, 9, 1, 1, 7, 1, 1, 1, 1, 7, 1, 1, 9, 9, 8],
    [9, 5, 1, 1, 7, 7, 1, 3, 3, 1, 7, 7, 1, 1, 5, 9],
    [9, 1, 6, 6, 1, 1, 3, 3, 3, 3, 1, 1, 6, 6, 1, 9],
    [1, 1, 6, 6, 1, 3, 3, 2, 2, 3, 3, 1, 6, 6, 1, 1],
    [1, 1, 6, 4, 4, 3, 3, 2, 2, 3, 3, 1, 6, 6, 1, 1],
    [9, 1, 6, 4, 4, 1, 3, 3, 3, 3, 1, 1, 6, 6, 1, 9],
    [9, 5, 1, 4, 4, 7, 1, 3, 3, 1, 7, 7, 1, 1, 5, 9],
    [8, 9, 9, 1, 1, 7, 1, 1, 1, 1, 7, 1, 1, 9, 9, 8],
    [6, 2, 3, 8, 1, 1, 6, 6, 6, 6, 1, 1, 8, 3, 2, 6],
    [7, 6, 3, 3, 9, 1, 6, 6, 6, 6, 1, 9, 3, 3, 6, 7],
    [2, 1, 6, 2, 9, 5, 1, 1, 1, 1, 5, 9, 2, 6, 1, 2],
    [2, 2, 7, 6, 8, 9, 9, 1, 1, 9, 9, 8, 6, 7, 2, 2],
], dtype=int)

E4_OUT = np.array([
    [2, 2, 7, 6, 8, 9, 9, 1, 1, 9, 9, 8, 6, 7, 2, 2],
    [2, 1, 6, 2, 9, 5, 1, 1, 1, 1, 5, 9, 2, 6, 1, 2],
    [7, 6, 3, 3, 9, 1, 6, 6, 6, 6, 1, 9, 3, 3, 6, 7],
    [6, 2, 3, 8, 1, 1, 6, 6, 6, 6, 1, 1, 8, 3, 2, 6],
    [8, 9, 9, 1, 1, 7, 1, 1, 1, 1, 7, 1, 1, 9, 9, 8],
    [9, 5, 1, 1, 7, 7, 1, 3, 3, 1, 7, 7, 1, 1, 5, 9],
    [9, 1, 6, 6, 1, 1, 3, 3, 3, 3, 1, 1, 6, 6, 1, 9],
    [1, 1, 6, 6, 1, 3, 3, 2, 2, 3, 3, 1, 6, 6, 1, 1],
    [1, 1, 6, 6, 1, 3, 3, 2, 2, 3, 3, 1, 6, 6, 1, 1],
    [9, 1, 6, 6, 1, 1, 3, 3, 3, 3, 1, 1, 6, 6, 1, 9],
    [9, 5, 1, 1, 7, 7, 1, 3, 3, 1, 7, 7, 1, 1, 5, 9],
    [8, 9, 9, 1, 1, 7, 1, 1, 1, 1, 7, 1, 1, 9, 9, 8],
    [6, 2, 3, 8, 1, 1, 6, 6, 6, 6, 1, 1, 8, 3, 2, 6],
    [7, 6, 3, 3, 9, 1, 6, 6, 6, 6, 1, 9, 3, 3, 6, 7],
    [2, 1, 6, 2, 9, 5, 1, 1, 1, 1, 5, 9, 2, 6, 1, 2],
    [2, 2, 7, 6, 8, 9, 9, 1, 1, 9, 9, 8, 6, 7, 2, 2],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [7, 7, 8, 1, 9, 8, 2, 6, 6, 2, 8, 9, 1, 8, 7, 7],
    [7, 1, 1, 8, 8, 8, 6, 6, 6, 6, 8, 8, 8, 1, 1, 7],
    [8, 1, 6, 9, 2, 6, 4, 4, 4, 6, 6, 2, 9, 6, 1, 8],
    [1, 8, 9, 1, 6, 6, 4, 4, 4, 1, 6, 6, 1, 9, 8, 1],
    [9, 8, 2, 6, 8, 7, 4, 4, 4, 6, 4, 4, 4, 4, 8, 9],
    [8, 8, 6, 6, 7, 7, 6, 5, 5, 6, 4, 4, 4, 4, 8, 8],
    [2, 6, 6, 1, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 6, 2],
    [6, 6, 1, 1, 6, 5, 5, 7, 7, 5, 4, 4, 4, 4, 6, 6],
    [6, 6, 1, 1, 6, 5, 5, 7, 7, 5, 5, 6, 1, 1, 6, 6],
    [2, 6, 6, 1, 6, 6, 5, 5, 5, 5, 6, 6, 1, 6, 6, 2],
    [8, 8, 6, 6, 7, 7, 6, 5, 5, 6, 7, 7, 6, 6, 8, 8],
    [9, 8, 2, 6, 8, 7, 6, 6, 6, 6, 7, 8, 6, 2, 8, 9],
    [1, 8, 9, 1, 6, 6, 1, 1, 1, 1, 6, 6, 1, 9, 8, 1],
    [8, 1, 6, 9, 2, 6, 6, 1, 1, 6, 6, 2, 9, 6, 1, 8],
    [7, 1, 1, 8, 8, 8, 6, 6, 6, 6, 8, 8, 8, 1, 1, 7],
    [7, 7, 8, 1, 9, 8, 2, 6, 6, 2, 8, 9, 1, 8, 7, 7],
], dtype=int)

T_OUT = np.array([
    [7, 7, 8, 1, 9, 8, 2, 6, 6, 2, 8, 9, 1, 8, 7, 7],
    [7, 1, 1, 8, 8, 8, 6, 6, 6, 6, 8, 8, 8, 1, 1, 7],
    [8, 1, 6, 9, 2, 6, 6, 1, 1, 6, 6, 2, 9, 6, 1, 8],
    [1, 8, 9, 1, 6, 6, 1, 1, 1, 1, 6, 6, 1, 9, 8, 1],
    [9, 8, 2, 6, 8, 7, 6, 6, 6, 6, 7, 8, 6, 2, 8, 9],
    [8, 8, 6, 6, 7, 7, 6, 5, 5, 6, 7, 7, 6, 6, 8, 8],
    [2, 6, 6, 1, 6, 6, 5, 5, 5, 5, 6, 6, 1, 6, 6, 2],
    [6, 6, 1, 1, 6, 5, 5, 7, 7, 5, 5, 6, 1, 1, 6, 6],
    [6, 6, 1, 1, 6, 5, 5, 7, 7, 5, 5, 6, 1, 1, 6, 6],
    [2, 6, 6, 1, 6, 6, 5, 5, 5, 5, 6, 6, 1, 6, 6, 2],
    [8, 8, 6, 6, 7, 7, 6, 5, 5, 6, 7, 7, 6, 6, 8, 8],
    [9, 8, 2, 6, 8, 7, 6, 6, 6, 6, 7, 8, 6, 2, 8, 9],
    [1, 8, 9, 1, 6, 6, 1, 1, 1, 1, 6, 6, 1, 9, 8, 1],
    [8, 1, 6, 9, 2, 6, 6, 1, 1, 6, 6, 2, 9, 6, 1, 8],
    [7, 1, 1, 8, 8, 8, 6, 6, 6, 6, 8, 8, 8, 1, 1, 7],
    [7, 7, 8, 1, 9, 8, 2, 6, 6, 2, 8, 9, 1, 8, 7, 7],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(g,L=len,R=range):
 h,w=L(g),L(g[0])
 for r in R(h//2+1):
  for c in R(w):
   if g[r][c]==4:g[r][c]=g[-(r+1)][c]
   if g[-(r+1)][c]==4:g[-(r+1)][c]=g[r][c]
 for r in R(h):
  for c in R(w//2+1):
   if g[r][c]==4:g[r][c]=g[r][-(c+1)]
   if g[r][-(c+1)]==4:g[r][-(c+1)]=g[r][c]
 return g


# --- Code Golf Solution (Compressed) ---
def q(*g):
    return g[g[0] == 4] * -1 * -1 or [*map(p, g[-1][::-1], *g)]


# --- RE-ARC Generator ---
from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable
from random import choice, sample, uniform

Integer = int

IntegerTuple = Tuple[Integer, Integer]

Grid = Tuple[Tuple[Integer]]

Cell = Tuple[Integer, IntegerTuple]

Object = FrozenSet[Cell]

Indices = FrozenSet[IntegerTuple]

Patch = Union[Object, Indices]

Piece = Union[Grid, Patch]

ContainerContainer = Container[Container]

def identity(
    x: Any
) -> Any:
    """ identity function """
    return x

def merge(
    containers: ContainerContainer
) -> Container:
    """ merging """
    return type(containers)(e for c in containers for e in c)

def sfilter(
    container: Container,
    condition: Callable
) -> Container:
    """ keep elements in container that satisfy condition """
    return type(container)(e for e in container if condition(e))

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

def ulcorner(
    patch: Patch
) -> IntegerTuple:
    """ index of upper left corner """
    return tuple(map(min, zip(*toindices(patch))))

def lrcorner(
    patch: Patch
) -> IntegerTuple:
    """ index of lower right corner """
    return tuple(map(max, zip(*toindices(patch))))

def toindices(
    patch: Patch
) -> Indices:
    """ indices of object cells """
    if len(patch) == 0:
        return frozenset()
    if isinstance(next(iter(patch))[1], tuple):
        return frozenset(index for value, index in patch)
    return patch

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

def asobject(
    grid: Grid
) -> Object:
    """ conversion of grid to object """
    return frozenset((v, (i, j)) for i, r in enumerate(grid) for j, v in enumerate(r))

def rot90(
    grid: Grid
) -> Grid:
    """ quarter clockwise rotation """
    return tuple(row for row in zip(*grid[::-1]))

def rot180(
    grid: Grid
) -> Grid:
    """ half rotation """
    return tuple(tuple(row[::-1]) for row in grid[::-1])

def rot270(
    grid: Grid
) -> Grid:
    """ quarter anticlockwise rotation """
    return tuple(tuple(row[::-1]) for row in zip(*grid[::-1]))[::-1]

def hmirror(
    piece: Piece
) -> Piece:
    """ mirroring along horizontal """
    if isinstance(piece, tuple):
        return piece[::-1]
    d = ulcorner(piece)[0] + lrcorner(piece)[0]
    if isinstance(next(iter(piece))[1], tuple):
        return frozenset((v, (d - i, j)) for v, (i, j) in piece)
    return frozenset((d - i, j) for i, j in piece)

def vmirror(
    piece: Piece
) -> Piece:
    """ mirroring along vertical """
    if isinstance(piece, tuple):
        return tuple(row[::-1] for row in piece)
    d = ulcorner(piece)[1] + lrcorner(piece)[1]
    if isinstance(next(iter(piece))[1], tuple):
        return frozenset((v, (i, d - j)) for v, (i, j) in piece)
    return frozenset((i, d - j) for i, j in piece)

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

def cmirror(
    piece: Piece
) -> Piece:
    """ mirroring along counterdiagonal """
    if isinstance(piece, tuple):
        return tuple(zip(*(r[::-1] for r in piece[::-1])))
    return vmirror(dmirror(vmirror(piece)))

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

def hconcat(
    a: Grid,
    b: Grid
) -> Grid:
    """ concatenate two grids horizontally """
    return tuple(i + j for i, j in zip(a, b))

def vconcat(
    a: Grid,
    b: Grid
) -> Grid:
    """ concatenate two grids vertically """
    return a + b

def canvas(
    value: Integer,
    dimensions: IntegerTuple
) -> Grid:
    """ grid construction """
    return tuple(tuple(value for j in range(dimensions[1])) for i in range(dimensions[0]))

def backdrop(
    patch: Patch
) -> Indices:
    """ indices in bounding box of patch """
    if len(patch) == 0:
        return frozenset({})
    indices = toindices(patch)
    si, sj = ulcorner(indices)
    ei, ej = lrcorner(patch)
    return frozenset((i, j) for i in range(si, ei + 1) for j in range(sj, ej + 1))

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

def generate_b8825c91(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(4, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (3, 15))
    w = h
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(diff_lb, diff_ub, (1, 8))
    remcols = sample(remcols, numcols)
    canv = canvas(bgc, (h, w))
    nc = unifint(diff_lb, diff_ub, (1, h * w))
    bx = asindices(canv)
    obj = {(choice(remcols), choice(totuple(bx)))}
    for kk in range(nc - 1):
        dns = mapply(neighbors, toindices(obj))
        ch = choice(totuple(bx & dns))
        obj.add((choice(remcols), ch))
        bx = bx - {ch}
    gi = paint(canv, obj)
    tr = sfilter(asobject(dmirror(gi)), lambda cij: cij[1][1] >= cij[1][0])
    gi = paint(gi, tr)
    gi = hconcat(gi, vmirror(gi))
    gi = vconcat(gi, hmirror(gi))
    go = tuple(e for e in gi)
    for alph in (2, 1):
        locidev = unifint(diff_lb, diff_ub, (1, alph*h))
        locjdev = unifint(diff_lb, diff_ub, (1, w))
        loci = alph*h - locidev
        locj = w - locjdev
        loci2 = unifint(diff_lb, diff_ub, (loci, alph*h - 1))
        locj2 = unifint(diff_lb, diff_ub, (locj, w - 1))
        bd = backdrop(frozenset({(loci, locj), (loci2, locj2)}))
        gi = fill(gi, 4, bd)
        gi, go = rot180(gi), rot180(go)
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = choice((1, 2))
    for fn in sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
IntegerSet = FrozenSet[Integer]

TupleTuple = Tuple[Tuple]

FOUR = 4

NEG_ONE = -1

def maximum(
    container: IntegerSet
) -> Integer:
    """ maximum """
    return max(container, default=0)

def pair(
    a: Tuple,
    b: Tuple
) -> TupleTuple:
    """ zipping of two tuples """
    return tuple(zip(a, b))

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

def papply(
    function: Callable,
    a: Tuple,
    b: Tuple
) -> Tuple:
    """ apply function on two vectors """
    return tuple(function(i, j) for i, j in zip(a, b))

def replace(
    grid: Grid,
    replacee: Integer,
    replacer: Integer
) -> Grid:
    """ color substitution """
    return tuple(tuple(replacer if v == replacee else v for v in r) for r in grid)

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_b8825c91(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = replace(I, FOUR, NEG_ONE)
    x1 = dmirror(x0)
    x2 = papply(pair, x0, x1)
    x3 = lbind(apply, maximum)
    x4 = apply(x3, x2)
    x5 = cmirror(x4)
    x6 = papply(pair, x4, x5)
    x7 = apply(x3, x6)
    x8 = hmirror(x7)
    x9 = papply(pair, x7, x8)
    x10 = apply(x3, x9)
    x11 = vmirror(x10)
    x12 = papply(pair, x11, x10)
    x13 = apply(x3, x12)
    return x13


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("E4", E4_IN, E4_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_b8825c91(inp)
        assert pred == _to_grid(expected), f"{name} failed"
