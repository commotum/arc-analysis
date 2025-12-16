"""
# [374] ea32f347.json
* separate_shapes
* count_tiles
* recoloring
* associate_colors_to_ranks
"""

p=lambda g:[[[g[i][j],g[1-i][j]][j%2]for j in range(6)]for i in range(2)]