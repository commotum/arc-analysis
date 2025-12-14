"""
# [216] 8efcae92.json
* separate_images
* rectangle_guessing
* count_tiles
* take_maximum
* crop
"""

p=lambda j:[[r for j,r in enumerate(j)if sum(r)and j%3==i%3][0]for i in range(len(j))]