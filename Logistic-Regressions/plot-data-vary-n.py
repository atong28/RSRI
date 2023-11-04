import json
import matplotlib.pyplot as plt
import numpy as np

INDICES = {0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0}
data = {}
for i in INDICES:
    data.update({str(i): [[] for _ in range(51)]})

for i in range(1, 6):
    with open(f'vary_n/data_{i}.json', 'r') as f:
        obj = json.load(f)
        for k, v in obj.items():
            for i in range(1, 51):
                data[k][i] += v[i]

l1 = [np.mean(row) if len(row) != 0 else 0 for row in data["1.0"]]
l2 = [np.mean(row) if len(row) != 0 else 0 for row in data["2.0"]]

line1, = plt.plot(list(range(500, 25001, 500)), l1[1:])
line2, = plt.plot(list(range(500, 25001, 500)), l2[1:])
plt.xlabel("n")
plt.ylabel("RMSE")
plt.legend([line1, line2], ["β=1.0", "β=2.0"])
plt.show()