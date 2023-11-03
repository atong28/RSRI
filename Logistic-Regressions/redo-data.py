import json

NUM = 5

for i in range(1, NUM+1):
    with open(f'vary_beta/data_{i}.json', 'r') as f:
        obj = json.load(f)
        
    for k in obj.keys():
        obj[k] = [i * 10 for i in obj[k]]
    with open(f'vary_beta/data_{i}.json', 'w') as f:
        f.write(json.dumps(obj))