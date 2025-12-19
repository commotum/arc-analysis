"""
# [334] d4469b4b.json
* dominant_color
* associate_images_to_colors
"""

def p(j):A={2:[[5,5,5],[0,5,0],[0,5,0]],1:[[0,5,0],[5,5,5],[0,5,0]],3:[[0,0,5],[0,0,5],[5,5,5]]};c=[i for s in j for i in s];return A[max(c)]