import json
import matplotlib.pyplot as plt
import numpy as np

INDICES = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]

data = {}
means = {}
for i in INDICES:
    data.update({str(i): [[] for _ in range(51)]})
    means.update({str(i): [[] for _ in range(51)]})

for i in range(1, 11):
    with open(f'vary_n/data_{i}.json', 'r') as f:
        obj = json.load(f)
        for k, v in obj.items():
            for i in range(1, 51):
                data[k][i] += v[i]

# find means
for k in data.keys():
    for i in range(1, 51):
        means[k] = [np.mean(row) if len(row) != 0 else 0 for row in data[k]]

lines = []
indices = []

# Uncomment to show one line
# '''
INDICES = [5.0]
x = np.linspace(500, 25000, 5000)
y = 21/((x * INDICES[0]) ** 0.50)
curve, = plt.plot(x, y)
lines.append(curve)
indices.append("Estimate")
# '''

for i in INDICES:
    if not means[str(i)][50]: continue
    line, = plt.plot(list(range(500, 25001, 500)), means[str(i)][1:])
    lines.append(line)
    indices.append(f"β={i}")

plt.xlabel("n")
plt.ylabel("RMSE")
plt.legend(lines, indices)
plt.show()