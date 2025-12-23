# --- Imports ---
import numpy as np

# --- Metadata ---
ARC_ID = "a8d7556c"
SERIAL = "265"
URL    = "https://arcprize.org/play?task=a8d7556c"

# --- Code Golf Concepts ---
CONCEPTS = [
    "recoloring",
    "rectangle_guessing",
]

# --- Example Grids ---
E1_IN = np.array([
    [5, 5, 5, 0, 5, 0, 0, 5, 5, 5, 5, 5, 5, 5, 0, 5, 5, 5],
    [5, 5, 0, 0, 0, 5, 0, 5, 0, 5, 5, 0, 0, 5, 0, 5, 0, 5],
    [0, 5, 5, 0, 5, 5, 0, 0, 5, 5, 0, 5, 5, 5, 5, 5, 0, 5],
    [5, 5, 0, 5, 5, 5, 5, 5, 5, 0, 5, 5, 5, 5, 5, 0, 5, 5],
    [5, 0, 5, 5, 5, 5, 5, 5, 5, 5, 0, 5, 5, 5, 0, 5, 0, 5],
    [0, 5, 5, 5, 5, 0, 0, 5, 0, 0, 5, 0, 5, 5, 5, 5, 5, 0],
    [0, 0, 5, 5, 5, 0, 0, 5, 0, 5, 0, 0, 0, 5, 5, 5, 5, 5],
    [0, 0, 5, 5, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 5],
    [5, 0, 5, 0, 5, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 0, 0, 5],
    [0, 0, 5, 5, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0, 5],
    [5, 5, 0, 5, 5, 5, 0, 0, 5, 0, 5, 0, 0, 5, 5, 5, 0, 5],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0, 5, 5, 5, 5, 5],
    [5, 5, 5, 5, 0, 0, 5, 5, 5, 5, 0, 5, 5, 0, 0, 5, 0, 0],
    [0, 5, 0, 0, 0, 5, 0, 5, 5, 0, 0, 5, 5, 5, 0, 0, 0, 5],
    [0, 0, 5, 5, 5, 5, 5, 0, 5, 0, 5, 0, 5, 0, 5, 5, 0, 0],
    [5, 0, 5, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 0, 0, 5, 0, 5],
    [5, 0, 5, 5, 0, 0, 0, 5, 5, 5, 0, 0, 0, 0, 0, 5, 0, 0],
    [5, 5, 0, 5, 0, 0, 5, 0, 0, 5, 5, 0, 5, 0, 5, 0, 5, 5],
], dtype=int)

E1_OUT = np.array([
    [5, 5, 5, 0, 5, 0, 0, 5, 5, 5, 5, 5, 5, 5, 0, 5, 5, 5],
    [5, 5, 0, 0, 0, 5, 0, 5, 0, 5, 5, 0, 0, 5, 0, 5, 0, 5],
    [0, 5, 5, 0, 5, 5, 0, 0, 5, 5, 0, 5, 5, 5, 5, 5, 0, 5],
    [5, 5, 0, 5, 5, 5, 5, 5, 5, 0, 5, 5, 5, 5, 5, 0, 5, 5],
    [5, 0, 5, 5, 5, 5, 5, 5, 5, 5, 0, 5, 5, 5, 0, 5, 0, 5],
    [0, 5, 5, 5, 5, 2, 2, 5, 0, 0, 5, 0, 5, 5, 5, 5, 5, 0],
    [2, 2, 5, 5, 5, 2, 2, 5, 0, 5, 0, 0, 0, 5, 5, 5, 5, 5],
    [2, 2, 5, 5, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2, 2, 5],
    [5, 0, 5, 0, 5, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 2, 2, 5],
    [0, 0, 5, 5, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0, 5],
    [5, 5, 0, 5, 5, 5, 0, 0, 5, 0, 5, 0, 0, 5, 5, 5, 0, 5],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0, 5, 5, 5, 5, 5],
    [5, 5, 5, 5, 0, 0, 5, 5, 5, 5, 0, 5, 5, 0, 0, 5, 0, 0],
    [0, 5, 0, 0, 0, 5, 0, 5, 5, 0, 0, 5, 5, 5, 0, 0, 0, 5],
    [0, 0, 5, 5, 5, 5, 5, 0, 5, 0, 5, 0, 5, 0, 5, 5, 0, 0],
    [5, 0, 5, 0, 2, 2, 5, 5, 5, 5, 5, 5, 5, 2, 2, 5, 0, 5],
    [5, 0, 5, 5, 2, 2, 0, 5, 5, 5, 0, 0, 0, 2, 2, 5, 0, 0],
    [5, 5, 0, 5, 2, 2, 5, 0, 0, 5, 5, 0, 5, 0, 5, 0, 5, 5],
], dtype=int)

E2_IN = np.array([
    [5, 5, 5, 5, 0, 5, 0, 5, 0, 5, 5, 5, 0, 0, 5, 0, 5, 5],
    [5, 5, 5, 5, 0, 0, 5, 5, 0, 5, 0, 0, 5, 0, 0, 5, 5, 0],
    [5, 5, 5, 5, 5, 5, 0, 5, 5, 5, 5, 0, 0, 0, 0, 5, 5, 0],
    [5, 0, 5, 5, 5, 5, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 0, 0],
    [0, 0, 5, 0, 5, 5, 0, 0, 0, 5, 5, 0, 0, 0, 5, 5, 5, 0],
    [5, 0, 0, 0, 5, 0, 5, 5, 5, 5, 0, 0, 0, 5, 0, 0, 0, 0],
    [0, 5, 0, 5, 5, 5, 0, 0, 0, 5, 5, 0, 0, 5, 0, 5, 5, 5],
    [5, 0, 0, 5, 5, 0, 5, 5, 0, 5, 0, 0, 5, 0, 5, 0, 5, 0],
    [5, 5, 5, 5, 0, 5, 5, 5, 0, 5, 5, 0, 5, 0, 5, 0, 5, 0],
    [5, 0, 5, 5, 5, 5, 0, 5, 0, 5, 0, 5, 5, 5, 0, 5, 5, 0],
    [5, 0, 5, 5, 5, 0, 5, 0, 5, 0, 0, 5, 0, 0, 5, 5, 5, 5],
    [0, 0, 0, 0, 5, 0, 5, 0, 0, 0, 5, 0, 5, 5, 5, 0, 0, 0],
    [5, 0, 5, 0, 0, 5, 0, 5, 5, 0, 0, 5, 0, 0, 0, 5, 5, 5],
    [5, 5, 5, 0, 5, 0, 0, 5, 5, 5, 0, 5, 5, 5, 0, 5, 5, 0],
    [0, 0, 5, 5, 5, 5, 5, 0, 5, 5, 5, 5, 0, 0, 0, 0, 0, 5],
    [5, 0, 5, 5, 5, 5, 0, 0, 5, 5, 0, 5, 0, 5, 5, 0, 5, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 0, 0, 0, 0, 5, 0, 0],
    [5, 5, 0, 0, 5, 5, 0, 5, 0, 5, 5, 5, 0, 5, 5, 5, 5, 5],
], dtype=int)

E2_OUT = np.array([
    [5, 5, 5, 5, 0, 5, 0, 5, 0, 5, 5, 5, 0, 0, 5, 0, 5, 5],
    [5, 5, 5, 5, 0, 0, 5, 5, 0, 5, 0, 0, 5, 2, 2, 5, 5, 0],
    [5, 5, 5, 5, 5, 5, 0, 5, 5, 5, 5, 0, 0, 2, 2, 5, 5, 0],
    [5, 0, 5, 5, 5, 5, 2, 2, 2, 0, 5, 5, 5, 5, 5, 5, 0, 0],
    [0, 0, 5, 0, 5, 5, 2, 2, 2, 5, 5, 2, 2, 0, 5, 5, 5, 0],
    [5, 0, 0, 0, 5, 0, 5, 5, 5, 5, 0, 2, 2, 5, 0, 0, 0, 0],
    [0, 5, 0, 5, 5, 5, 0, 0, 0, 5, 5, 2, 2, 5, 0, 5, 5, 5],
    [5, 0, 0, 5, 5, 0, 5, 5, 0, 5, 0, 0, 5, 0, 5, 0, 5, 0],
    [5, 5, 5, 5, 0, 5, 5, 5, 0, 5, 5, 0, 5, 0, 5, 0, 5, 0],
    [5, 0, 5, 5, 5, 5, 0, 5, 0, 5, 0, 5, 5, 5, 0, 5, 5, 0],
    [5, 0, 5, 5, 5, 0, 5, 0, 5, 0, 0, 5, 0, 0, 5, 5, 5, 5],
    [0, 0, 0, 0, 5, 0, 5, 0, 0, 0, 5, 0, 5, 5, 5, 0, 0, 0],
    [5, 0, 5, 0, 0, 5, 0, 5, 5, 0, 0, 5, 0, 0, 0, 5, 5, 5],
    [5, 5, 5, 0, 5, 0, 0, 5, 5, 5, 0, 5, 5, 5, 0, 5, 5, 0],
    [0, 0, 5, 5, 5, 5, 5, 0, 5, 5, 5, 5, 0, 0, 0, 0, 0, 5],
    [5, 0, 5, 5, 5, 5, 2, 2, 5, 5, 0, 5, 0, 5, 5, 0, 5, 0],
    [0, 0, 2, 2, 0, 0, 2, 2, 5, 5, 5, 0, 0, 0, 0, 5, 0, 0],
    [5, 5, 2, 2, 5, 5, 0, 5, 0, 5, 5, 5, 0, 5, 5, 5, 5, 5],
], dtype=int)

E3_IN = np.array([
    [0, 0, 5, 5, 5, 5, 5, 5, 5, 0, 0, 5, 5, 5, 0, 5, 5, 0],
    [5, 0, 0, 0, 5, 5, 0, 0, 0, 0, 5, 0, 5, 5, 0, 5, 5, 5],
    [0, 0, 5, 5, 5, 5, 0, 0, 5, 5, 5, 5, 0, 0, 0, 5, 5, 5],
    [5, 5, 5, 0, 5, 5, 5, 5, 5, 5, 0, 0, 5, 5, 5, 5, 5, 5],
    [5, 5, 0, 5, 5, 5, 5, 0, 5, 5, 5, 5, 0, 5, 0, 0, 0, 0],
    [5, 0, 0, 5, 5, 5, 5, 5, 5, 0, 5, 5, 5, 0, 5, 0, 0, 5],
    [5, 5, 5, 0, 5, 5, 5, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 0],
    [0, 5, 5, 0, 5, 5, 5, 5, 0, 5, 0, 0, 5, 0, 5, 5, 5, 0],
    [5, 5, 5, 5, 5, 0, 5, 5, 0, 5, 0, 0, 0, 5, 0, 5, 0, 5],
    [5, 0, 5, 0, 5, 0, 5, 5, 5, 5, 0, 0, 0, 5, 5, 5, 5, 5],
    [0, 0, 5, 0, 5, 5, 0, 5, 5, 5, 0, 0, 5, 0, 5, 5, 5, 5],
    [5, 5, 5, 5, 0, 5, 5, 5, 5, 0, 5, 5, 5, 5, 5, 0, 5, 5],
    [0, 0, 5, 5, 5, 0, 5, 5, 0, 5, 5, 0, 5, 0, 5, 5, 5, 5],
    [5, 5, 0, 5, 5, 5, 0, 0, 0, 0, 5, 0, 5, 5, 0, 5, 0, 0],
    [0, 0, 5, 5, 5, 5, 0, 5, 5, 0, 5, 0, 0, 0, 5, 0, 5, 0],
    [0, 5, 5, 5, 5, 5, 0, 5, 5, 5, 0, 5, 0, 5, 5, 0, 0, 5],
    [0, 5, 5, 0, 0, 5, 5, 5, 0, 0, 0, 5, 5, 0, 5, 5, 5, 5],
    [5, 0, 0, 5, 5, 0, 5, 5, 5, 5, 5, 0, 5, 5, 0, 0, 5, 0],
], dtype=int)

E3_OUT = np.array([
    [0, 0, 5, 5, 5, 5, 5, 5, 5, 0, 0, 5, 5, 5, 0, 5, 5, 0],
    [5, 0, 0, 0, 5, 5, 2, 2, 0, 0, 5, 0, 5, 5, 0, 5, 5, 5],
    [0, 0, 5, 5, 5, 5, 2, 2, 5, 5, 5, 5, 0, 0, 0, 5, 5, 5],
    [5, 5, 5, 0, 5, 5, 5, 5, 5, 5, 0, 0, 5, 5, 5, 5, 5, 5],
    [5, 5, 0, 5, 5, 5, 5, 0, 5, 5, 5, 5, 0, 5, 0, 2, 2, 0],
    [5, 0, 0, 5, 5, 5, 5, 5, 5, 0, 5, 5, 5, 0, 5, 2, 2, 5],
    [5, 5, 5, 0, 5, 5, 5, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 0],
    [0, 5, 5, 0, 5, 5, 5, 5, 0, 5, 2, 2, 5, 0, 5, 5, 5, 0],
    [5, 5, 5, 5, 5, 0, 5, 5, 0, 5, 2, 2, 0, 5, 0, 5, 0, 5],
    [5, 0, 5, 0, 5, 0, 5, 5, 5, 5, 2, 2, 0, 5, 5, 5, 5, 5],
    [0, 0, 5, 0, 5, 5, 0, 5, 5, 5, 2, 2, 5, 0, 5, 5, 5, 5],
    [5, 5, 5, 5, 0, 5, 5, 5, 5, 0, 5, 5, 5, 5, 5, 0, 5, 5],
    [0, 0, 5, 5, 5, 0, 5, 5, 0, 5, 5, 0, 5, 0, 5, 5, 5, 5],
    [5, 5, 0, 5, 5, 5, 0, 0, 0, 0, 5, 0, 5, 5, 0, 5, 0, 0],
    [0, 0, 5, 5, 5, 5, 0, 5, 5, 0, 5, 0, 0, 0, 5, 0, 5, 0],
    [0, 5, 5, 5, 5, 5, 0, 5, 5, 5, 0, 5, 0, 5, 5, 0, 0, 5],
    [0, 5, 5, 0, 0, 5, 5, 5, 0, 0, 0, 5, 5, 0, 5, 5, 5, 5],
    [5, 0, 0, 5, 5, 0, 5, 5, 5, 5, 5, 0, 5, 5, 0, 0, 5, 0],
], dtype=int)

# --- Test ---
T_IN = np.array([
    [0, 0, 0, 5, 0, 5, 0, 0, 5, 5, 0, 5, 5, 5, 5, 5, 0, 0],
    [0, 0, 5, 5, 0, 5, 0, 5, 0, 0, 0, 5, 5, 5, 5, 0, 5, 5],
    [5, 0, 0, 0, 5, 5, 0, 5, 0, 0, 5, 0, 5, 0, 5, 5, 0, 5],
    [0, 5, 5, 5, 0, 5, 5, 0, 5, 5, 0, 0, 0, 5, 5, 0, 5, 5],
    [5, 5, 5, 5, 5, 5, 5, 0, 0, 5, 5, 0, 0, 0, 0, 5, 5, 5],
    [0, 5, 5, 5, 5, 0, 5, 5, 5, 0, 5, 0, 0, 5, 5, 0, 5, 0],
    [5, 5, 0, 5, 5, 5, 5, 5, 5, 0, 0, 5, 0, 0, 5, 0, 5, 5],
    [5, 5, 5, 5, 0, 0, 5, 5, 0, 5, 5, 5, 5, 5, 0, 5, 5, 0],
    [5, 0, 5, 0, 0, 5, 5, 5, 0, 0, 0, 5, 5, 5, 5, 0, 0, 0],
    [0, 0, 0, 0, 5, 0, 0, 0, 5, 5, 5, 5, 0, 0, 5, 0, 5, 5],
    [5, 0, 5, 5, 0, 5, 5, 5, 0, 0, 5, 0, 5, 5, 5, 5, 5, 0],
    [0, 0, 0, 5, 5, 0, 5, 0, 0, 5, 5, 0, 5, 5, 5, 5, 5, 5],
    [0, 5, 5, 5, 5, 0, 0, 5, 0, 0, 0, 5, 5, 5, 5, 5, 0, 5],
    [5, 5, 5, 5, 5, 5, 5, 0, 5, 5, 5, 5, 5, 5, 5, 0, 0, 5],
    [5, 5, 0, 5, 5, 5, 0, 5, 0, 5, 5, 5, 5, 0, 5, 0, 0, 5],
    [5, 0, 5, 5, 5, 5, 0, 5, 5, 0, 0, 0, 5, 5, 5, 5, 0, 5],
    [0, 5, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 0, 0, 0, 0, 0, 0, 5, 0, 5, 0, 5, 5, 0, 5, 5, 0, 0],
], dtype=int)

T_OUT = np.array([
    [2, 2, 0, 5, 0, 5, 0, 0, 5, 5, 0, 5, 5, 5, 5, 5, 0, 0],
    [2, 2, 5, 5, 0, 5, 0, 5, 2, 2, 0, 5, 5, 5, 5, 0, 5, 5],
    [5, 0, 0, 0, 5, 5, 0, 5, 2, 2, 5, 0, 5, 0, 5, 5, 0, 5],
    [0, 5, 5, 5, 0, 5, 5, 0, 5, 5, 0, 2, 2, 5, 5, 0, 5, 5],
    [5, 5, 5, 5, 5, 5, 5, 0, 0, 5, 5, 2, 2, 0, 0, 5, 5, 5],
    [0, 5, 5, 5, 5, 0, 5, 5, 5, 0, 5, 2, 2, 5, 5, 0, 5, 0],
    [5, 5, 0, 5, 5, 5, 5, 5, 5, 0, 0, 5, 0, 0, 5, 0, 5, 5],
    [5, 5, 5, 5, 0, 0, 5, 5, 0, 5, 5, 5, 5, 5, 0, 5, 5, 0],
    [5, 0, 5, 0, 0, 5, 5, 5, 0, 0, 0, 5, 5, 5, 5, 0, 0, 0],
    [0, 0, 0, 0, 5, 0, 0, 0, 5, 5, 5, 5, 0, 0, 5, 0, 5, 5],
    [5, 0, 5, 5, 0, 5, 5, 5, 0, 0, 5, 0, 5, 5, 5, 5, 5, 0],
    [0, 0, 0, 5, 5, 0, 5, 0, 0, 5, 5, 0, 5, 5, 5, 5, 5, 5],
    [0, 5, 5, 5, 5, 0, 0, 5, 0, 0, 0, 5, 5, 5, 5, 5, 0, 5],
    [5, 5, 5, 5, 5, 5, 5, 0, 5, 5, 5, 5, 5, 5, 5, 2, 2, 5],
    [5, 5, 0, 5, 5, 5, 0, 5, 0, 5, 5, 5, 5, 0, 5, 2, 2, 5],
    [5, 0, 5, 5, 5, 5, 0, 5, 5, 0, 0, 0, 5, 5, 5, 5, 0, 5],
    [0, 5, 2, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 0, 2, 2, 0, 0, 0, 5, 0, 5, 0, 5, 5, 0, 5, 5, 0, 0],
], dtype=int)

# --- Code Golf Solution (Barnacles) ---
def p(g,L=len,R=range):
 h,w=L(g),L(g[0])
 for r in R(h-1):
  for c in R(w-1):
   C=g[r][c:c+2]+g[r+1][c:c+2]
   if C.count(0)==4:
    g[r][c]=2
    g[r][c+1]=2
    g[r+1][c]=2
    g[r+1][c+1]=2
   if C.count(0)==2 and C.count(2)==2:
    g[r][c]=2
    g[r][c+1]=2
    g[r+1][c]=2
    g[r+1][c+1]=2
 return g


# --- Code Golf Solution (Compressed) ---
def q(g):
    return eval(re.sub('0(?=.{949,952}(.{56})?0(?!.{37}0.{485}]), 0.{52}0, 0)', '2', '%r#' % g * 2))


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

def difference(
    a: Container,
    b: Container
) -> Container:
    """ difference """
    return type(a)(e for e in a if e not in b)

def totuple(
    container: FrozenSet
) -> Tuple:
    """ conversion to tuple """
    return tuple(container)

def interval(
    start: Integer,
    stop: Integer,
    step: Integer
) -> Tuple:
    """ range """
    return tuple(range(start, stop, step))

def asindices(
    grid: Grid
) -> Indices:
    """ indices of all grid cells """
    return frozenset((i, j) for i in range(len(grid)) for j in range(len(grid[0])))

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

def generate_a8d7556c(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (0, 2))
    h = unifint(diff_lb, diff_ub, (6, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    fgc = choice(cols)
    c = canvas(fgc, (h, w))
    numblacks = unifint(diff_lb, diff_ub, (1, (h * w) // 3 * 2))
    inds = totuple(asindices(c))
    blacks = sample(inds, numblacks)
    gi = fill(c, 0, blacks)
    numsq = unifint(diff_lb, diff_ub, (1, (h * w) // 10))
    sqlocs = sample(inds, numsq)
    gi = fill(gi, 0, shift(sqlocs, (0, 0)))
    gi = fill(gi, 0, shift(sqlocs, (0, 1)))
    gi = fill(gi, 0, shift(sqlocs, (1, 0)))
    gi = fill(gi, 0, shift(sqlocs, (1, 1)))
    go = tuple(e for e in gi)
    for a in range(h - 1):
        for b in range(w - 1):
            if gi[a][b] == 0 and gi[a+1][b] == 0 and gi[a][b+1] == 0 and gi[a+1][b+1] == 0:
                go = fill(go, 2, {(a, b), (a+1, b), (a, b+1), (a+1, b+1)})
    return {'input': gi, 'output': go}


# --- RE-ARC Verifier ---
Element = Union[Object, Grid]

ContainerContainer = Container[Container]

ZERO = 0

TWO = 2

ORIGIN = (0, 0)

def merge(
    containers: ContainerContainer
) -> Container:
    """ merging """
    return type(containers)(e for c in containers for e in c)

def initset(
    value: Any
) -> FrozenSet:
    """ initialize container """
    return frozenset({value})

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

def ulcorner(
    patch: Patch
) -> IntegerTuple:
    """ index of upper left corner """
    return tuple(map(min, zip(*toindices(patch))))

def recolor(
    value: Integer,
    patch: Patch
) -> Object:
    """ recolor patch """
    return frozenset((value, index) for index in toindices(patch))

def normalize(
    patch: Patch
) -> Patch:
    """ moves upper left corner to origin """
    if len(patch) == 0:
        return patch
    return shift(patch, (-uppermost(patch), -leftmost(patch)))

def uppermost(
    patch: Patch
) -> Integer:
    """ row index of uppermost occupied cell """
    return min(i for i, j in toindices(patch))

def leftmost(
    patch: Patch
) -> Integer:
    """ column index of leftmost occupied cell """
    return min(j for i, j in toindices(patch))

def upscale(
    element: Element,
    factor: Integer
) -> Element:
    """ upscale object or grid """
    if isinstance(element, tuple):
        upscaled_grid = tuple()
        for row in element:
            upscaled_row = tuple()
            for value in row:
                upscaled_row = upscaled_row + tuple(value for num in range(factor))
            upscaled_grid = upscaled_grid + tuple(upscaled_row for num in range(factor))
        return upscaled_grid
    else:
        if len(element) == 0:
            return frozenset()
        di_inv, dj_inv = ulcorner(element)
        di, dj = (-di_inv, -dj_inv)
        normed_obj = shift(element, (di, dj))
        upscaled_obj = set()
        for value, (i, j) in normed_obj:
            for io in range(factor):
                for jo in range(factor):
                    upscaled_obj.add((value, (i * factor + io, j * factor + jo)))
        return shift(frozenset(upscaled_obj), (di_inv, dj_inv))

def occurrences(
    grid: Grid,
    obj: Object
) -> Indices:
    """ locations of occurrences of object in grid """
    occurrences = set()
    normed = normalize(obj)
    h, w = len(grid), len(grid[0])
    for i in range(h):
        for j in range(w):
            occurs = True
            for v, (a, b) in shift(normed, (i, j)):
                if 0 <= a < h and 0 <= b < w:
                    if grid[a][b] != v:
                        occurs = False
                        break
                else:
                    occurs = False
                    break
            if occurs:
                occurrences.add((i, j))
    return frozenset(occurrences)

def _to_grid(
    grid: Any
) -> Grid:
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(x) for x in row) for row in grid)
    return grid

def verify_a8d7556c(I: Grid) -> Grid:
    I = _to_grid(I)
    x0 = initset(ORIGIN)
    x1 = recolor(ZERO, x0)
    x2 = upscale(x1, TWO)
    x3 = occurrences(I, x2)
    x4 = lbind(shift, x2)
    x5 = mapply(x4, x3)
    x6 = fill(I, TWO, x5)
    return x6


if __name__ == "__main__":
    examples = [
        ("E1", E1_IN, E1_OUT),
        ("E2", E2_IN, E2_OUT),
        ("E3", E3_IN, E3_OUT),
        ("T", T_IN, T_OUT),
    ]
    for name, inp, expected in examples:
        pred = verify_a8d7556c(inp)
        assert pred == _to_grid(expected), f"{name} failed"
