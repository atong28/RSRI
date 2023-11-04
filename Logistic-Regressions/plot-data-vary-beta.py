import json
import matplotlib.pyplot as plt
import numpy as np

INDICES = {"500","1000","2500","5000","10000","15000","20000","25000"}
data = {}
for i in INDICES:
    data.update({i: [[] for _ in range(101)]})

for i in range(1, 11):
    with open(f'vary_beta/data_{i}.json', 'r') as f:
        obj = json.load(f)
        for k, v in obj.items():
            for i in range(1, 101):
                data[k][i] += v[i]

l500 = [np.mean(row) if len(row) != 0 else 0 for row in data["500"]]
l1000 = [np.mean(row) if len(row) != 0 else 0 for row in data["1000"]]
l2500 = [np.mean(row) if len(row) != 0 else 0 for row in data["2500"]]
l5000 = [np.mean(row) if len(row) != 0 else 0 for row in data["5000"]]
l10000 = [np.mean(row) if len(row) != 0 else 0 for row in data["10000"]]
l15000 = [np.mean(row) if len(row) != 0 else 0 for row in data["15000"]]
l20000 = [np.mean(row) if len(row) != 0 else 0 for row in data["20000"]]
l25000 = [np.mean(row) if len(row) != 0 else 0 for row in data["25000"]]

line500, = plt.plot(list(range(1, 101)), l500[1:])
line1000, = plt.plot(list(range(1, 101)), l1000[1:])
line2500, = plt.plot(list(range(1, 101)), l2500[1:])
line5000, = plt.plot(list(range(1, 101)), l5000[1:])
line10000, = plt.plot(list(range(1, 101)), l10000[1:])
line15000, = plt.plot(list(range(1, 101)), l15000[1:])
line20000, = plt.plot(list(range(1, 101)), l20000[1:])
line25000, = plt.plot(list(range(1, 101)), l25000[1:])

plt.xlabel("Beta")
plt.ylabel("RMSE")
plt.legend([line500, line1000, line2500, line5000, line10000, line15000, line20000, line25000], ["n=500","n=1000","n=2500", "n=5000", "n=10000", "n=15000", "n=20000", "n=25000"])
plt.show()