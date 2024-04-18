import matplotlib.pyplot as plt
import json

if __name__ == '__main__':
    data = []
    for iter in range(1, 2):
        with open(f'vary_n/{iter}.txt', 'r') as f:
            data += json.load(f)
    data = sorted(data, key=lambda x: x[0])
    n = [entry[0] for entry in data]
    p_acc = [entry[2] for entry in data]
    l_acc = [entry[3] for entry in data]
    
    lines = []
    indices = []
    
    p_line, = plt.plot(n, p_acc)
    l_line, = plt.plot(n, l_acc)
    
    plt.xlabel("n")
    plt.ylabel("RMSE")
    plt.legend([p_line, l_line], ['Perceptron', 'Logistic'])
    plt.show()
            
            