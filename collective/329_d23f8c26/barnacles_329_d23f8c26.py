"""
# [330] d2abd087.json
* separate_shapes
* count_tiles
* associate_colors_to_numbers
* recoloring
"""

def p(j):
	A=len(j[0])//2;c=[[0 for A in A]for A in j]
	for E in range(len(j)):c[E][A]=j[E][A]
	return c