"""
# [236] 99b1bc43.json
* take_complement
* detect_wall
* separate_images
* pattern_intersection
"""

p=lambda j:[[(45-j[2][x]-2*j[2][x+1]-4*j[1][x+1])//5]*3 for x in range(0,15,5)]