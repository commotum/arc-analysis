"""
# [229] 9565186b.json
* separate_shapes
* count_tiles
* recoloring
* take_maximum
* associate_color_to_bools
"""

def p(j):A=__import__('collections').Counter([x for R in j for x in R]).most_common(1);c=A[0][0];return[[A if A==c else 5 for A in R]for R in j]