"""
# [145] 6455b5f5.json
* measure_area
* take_maximum
* take_minimum
* loop_filling
* associate_colors_to_ranks
"""

p=lambda g:[[3if g[i][j]==0and g[i+5][j]==0else 0for j in range(4)]for i in range(4)]