"""
# [054] 264363fd.json
* pattern_repetition
* pattern_juxtaposition
* draw_line_from_point
"""

p=lambda j:[r[3%len(r):]+r[:3%len(r)]for r in j[2:]+j[:2]]