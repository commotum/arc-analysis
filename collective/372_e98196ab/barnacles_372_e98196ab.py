"""
# [372] e98196ab.json
* detect_wall
* separate_images
* image_juxtaposition
"""

p=lambda g:[[g[i][j]or g[i+6][j]for j in range(11)]for i in range(5)]