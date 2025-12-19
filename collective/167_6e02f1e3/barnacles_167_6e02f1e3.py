"""
# [167] 6e02f1e3.json
* count_different_colors
* associate_images_to_numbers
"""

p=lambda j:[[[5,5,5],[0,0,0],[0,0,0]],[[5,0,0],[0,5,0],[0,0,5]],[[0,0,5],[0,5,0],[5,0,0]]][len(set(v for r in j for v in r))-1]