import matplotlib.pyplot as plt
import json

if __name__ == '__main__':
    data = []
    for iter in range(1, 2):
        with open(f'vary_beta/{iter}.txt', 'r') as f:
            data += json.load(f)
    data = sorted(data, key=lambda x: x[1])
    b = [entry[1] for entry in data]
    p_acc = [entry[2] for entry in data]
    l_acc = [entry[3] for entry in data]
    
    lines = []
    indices = []
    
    p_line, = plt.plot(b, p_acc)
    l_line, = plt.plot(b, l_acc)
    
    plt.xlabel("beta")
    plt.ylabel("RMSE")
    plt.legend([p_line, l_line], ['Perceptron', 'Logistic'])
    plt.show()
            
            