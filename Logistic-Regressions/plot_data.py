import json
import matplotlib.pyplot as plt
import numpy as np

with open('data.json', 'r') as f:
    obj = json.load(f)

l = [np.mean(row) if len(row) != 0 else 0 for row in obj["25000"]]

plt.plot(list(range(1, 101)), l[1:])
plt.xlabel("Beta")
plt.ylabel("RMSE")
plt.show()