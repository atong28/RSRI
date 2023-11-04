import json

# CAN POTENTIALLY ERASE DATA, RUN WITH CARE

obj = {}

typeBeta = False

INDICES = [500, 1000, 2500, 5000, 10000, 15000, 20000, 25000]
N_INDICES = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]

if typeBeta:
    for i in INDICES:
        obj.update({str(i): [[] for _ in range(101)]})
    for i in range(11, 21):
        with open(f'vary_beta/data_{i}.json', 'w') as f:
            f.write(json.dumps(obj))
else:
    for i in N_INDICES:
        obj.update({str(i): [[] for _ in range(51)]})
    for i in range(11, 21):
        with open(f'vary_n/data_{i}.json', 'w') as f:
            f.write(json.dumps(obj))