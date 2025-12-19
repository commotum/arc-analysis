"""
# [235] 995c5fa3.json
* take_complement
* detect_wall
* separate_images
* associate_colors_to_images
* summarize
"""

p=lambda j:[[(45-j[2][x]-2*j[2][x+1]-4*j[1][x+1])//5]*3 for x in range(0,15,5)]