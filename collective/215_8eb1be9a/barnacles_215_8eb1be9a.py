"""
# [215] 8eb1be9a.json
* pattern_repetition
* image_filling
"""

p=lambda j:[[r for j,r in enumerate(j)if sum(r)and j%3==i%3][0]for i in range(len(j))]