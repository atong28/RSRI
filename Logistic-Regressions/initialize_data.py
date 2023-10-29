import json

# CAN POTENTIALLY ERASE DATA, RUN WITH CARE

NUM = 5

obj = {}

INDICES = {"500","1000","2500","5000","10000","15000","20000","25000"}

for i in INDICES:
    obj.update({i: [[] for _ in range(101)]})

with open(f'data_{NUM}.json', 'w') as f:
    f.write(json.dumps(obj))