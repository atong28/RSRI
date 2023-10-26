import json

with open('data.json', 'r') as f:
    obj = json.load(f)

obj.update({"500": [[] for _ in range(101)]})

with open('data.json', 'w') as f:
    f.write(json.dumps(obj))