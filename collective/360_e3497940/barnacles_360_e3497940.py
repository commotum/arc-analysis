"""
# [360] e3497940.json
* detect_wall
* separate_images
* image_reflection
* image_juxtaposition
"""

p=lambda g:[[g[i][j]or g[i][8-j]if g[i][j]*g[i][8-j]==0 else g[i][j]for j in range(4)]for i in range(len(g))]