import json
import numpy as np

INDICES = {0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0}
data = {}
means = {}
for i in INDICES:
    data.update({str(i): [[] for _ in range(51)]})
    means.update({str(i): [[] for _ in range(51)]})

# compile together
for i in range(1, 21):
    with open(f'vary_n/data_{i}.json', 'r') as f:
        obj = json.load(f)
        for k, v in obj.items():
            for i in range(1, 51):
                data[k][i] += v[i]
                         
# find means
for k in data.keys():
    for i in range(1, 51):
        means[k] = [np.mean(row) if len(row) != 0 else 0 for row in data[k]]
        
with open(f'vary_n/data.json', 'w') as f:
    f.write(json.dumps(data))