"""
# [391] f8b3ba0a.json
* detect_grid
* find_the_intruder
* dominant_color
* count_tiles
* summarize
* order_numbers
"""

p=lambda j:[[k]for k,_ in __import__('collections').Counter(i for r in j for i in r).most_common(5)[2:]]