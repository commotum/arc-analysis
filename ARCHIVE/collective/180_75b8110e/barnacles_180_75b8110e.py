"""
# [180] 75b8110e.json
* separate_images
* image_juxtaposition
"""

p=lambda j,A=range(4):[[j[x][y+4]or j[x+4][y]or j[x+4][y+4]or j[x][y]for y in A]for x in A]