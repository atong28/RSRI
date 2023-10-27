import json

# CAN POTENTIALLY ERASE DATA, RUN WITH CARE

with open('data.json', 'r') as f:
    obj = json.load(f)

obj.update({"20000": [[] for _ in range(101)]})

with open('data.json', 'w') as f:
    f.write(json.dumps(obj))