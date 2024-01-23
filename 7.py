import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5,5,100).reshape(-1,1)
y = np.sin(x).ravel() + 0.1 * np.random.randn(100)

def rbf_ker_fun(x,c,s):
    rbf = np.exp(-0.5 * ((x-c)/s)**2)
    return rbf

def train():
    centers = np.linspace(-5,5,10)
    widths = 1.0
    
    rbf = rbf_ker_fun(x,centers,widths)
    phi = np.column_stack([rbf for c in centers])
    
    weights = np.linalg.lstsq(phi,y, rcond=None)[0]
    y_pred = phi.dot(weights)
    
    return y_pred

y_pred = train()

plt.scatter(x,y,label='Training datapoints')
plt.plot(x,y_pred,color='red', label='predicted')
plt.xlabel('input')
plt.ylabel('output')
plt.title("Regression using RBF")
plt.legend()
plt.show()

