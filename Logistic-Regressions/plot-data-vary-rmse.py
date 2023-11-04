import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

index = "1.0"
epsilon = 1/float(index)

with open(f'vary_RMSE/data_{index}.json', 'r') as f:
    obj = json.load(f)

n_list = []
means_list = []

CENTER = 460
DELTA = 10

# find means
for k in obj.keys():
    # set bound
    if not (CENTER - DELTA <= int(k) <= CENTER+DELTA): continue
    n_list.append(int(k))
    means_list.append(np.mean(obj[k]))

sorted_mean_list = [x for _, x in sorted(zip(n_list, means_list))]
sorted_n_list = sorted(n_list)

print(np.array(sorted_n_list).reshape(-1, 1))
print(np.array(sorted_mean_list).reshape(1, -1))

reg = LinearRegression().fit(np.array(sorted_n_list).reshape(-1, 1), np.array(sorted_mean_list))
x = np.linspace(CENTER - DELTA, CENTER+DELTA, 2*DELTA)
print(reg.coef_)
print(reg.intercept_)
y = reg.coef_[0] * x + reg.intercept_
bestfit, = plt.plot(x, y)

line, = plt.plot(sorted_n_list, sorted_mean_list)
line_base, = plt.plot(n_list, [epsilon] * (len(n_list)))

plt.xlabel("n")
plt.ylabel("RMSE")
plt.legend([line, line_base, bestfit], [f"β={index}", f"ϵ={epsilon}", "Best Fit"])
plt.show()