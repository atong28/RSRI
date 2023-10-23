import json

obj = [[] for _ in range(101)]

with open('data.json', 'w') as f:
    f.write(json.dumps(obj))