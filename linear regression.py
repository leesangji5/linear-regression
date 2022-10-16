import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('linear regression\data.csv', delimiter=',')
x = data[:, 0]
y = data[:, 1]

w = 0
b = 0
lrate = 0.01
epochs = 1000

n = float(len(x))

for i in range(epochs):
    y_pred = w * x + b
    D_w = (2/n) * sum(x * (y_pred - y))
    D_b = (2/n) * sum(y_pred - y)
    w = w - lrate * D_w
    b = b - lrate * D_b

print(w, b)

y_pred = w * x + b

plt.scatter(x, y)
plt.plot([min(x), max(x)], [min(y_pred), max(y_pred)], color='red')
plt.show()