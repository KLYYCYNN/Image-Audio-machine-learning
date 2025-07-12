
import numpy as np
import matplotlib.pyplot as plt

# %%

np.set_printoptions(precision=4)


# %%

def func1(x): # any function for training data generation
    return np.cos(5*x) + np.sin(3*x) - np.exp(-x**2/11)

def func3(x):
    return np.cos(x)
def func4(x):
    return np.exp(- x**(-0.4)) * (5*np.cos(3*x) + 7*np.sin(2*x))

#Creates a random array of 1 and -1
def pm_arr(n):
    pm = [-1, 1]
    return np.random.choice(pm, n)

def training_set_gen(f, xmin, xmax, n):
    x = np.linspace(xmin, xmax, 5*n)
    del_ind = np.random.choice(5*n, 4*n, replace = False) # creates 5n x-values and randomly deletes 4n of them, leaving only n
    x = np.delete(x, del_ind)
    
    y = f(x)
    d = np.ptp(y)
    y_flu = d * 0.1 * np.random.rand(n) * pm_arr(n) #random fluctuations, with maximum amplitude d/5
    y += y_flu
    
    return x, y

def feature_translation(x):
    return x - min(x) - np.ptp(x)/2


def Z_score_normalization(x):
    return (x - np.mean(x)) / np.std(x)

def scale_params(unscaled_feature, w, b, op):
    mean, std = np.array( [np.mean(x) for x in unscaled_feature] ), np.array( [np.std(x) for x in unscaled_feature] )
    
    assert op == 'scale' or op == 'undo' , 'unspecified operation'
    if op == 'scale':
        w_scaled = w * std
        b_scaled = b + np.dot(w, mean)
    else:
        w_scaled = w / std
        b_scaled = b - np.dot(w_scaled, mean)
    
    return w_scaled, b_scaled
            
#somehow this doesn't work well with large |x|. Turning on the feature translation can make the fit look good
#but parameters are not actually correct
def polynomial_regression(x, y, order, alpha, n, feature_translation = False):
    
    X, X_unscaled = np.zeros((order, len(x))), np.zeros((order, len(x)))
    if feature_translation == True:
        x = feature_translation(x) 
    for i in range(order): #create feature with higher orders of x
        X[i], X_unscaled[i] = Z_score_normalization(x ** (i+1)), x ** (i+1)

    cost = []
    w, b = np.zeros(order), 0
    for i in range(n):
        
        y_hat = np.zeros(len(y))
        for j in range(len(x)):          
            y_hat[j] = np.dot(w, X[:, j])
        y_hat += b
        pulls = y_hat - y # compute pulls at all data points using w dot x^(i)
        
        w_update = np.zeros(len(w))
        for j in range(order):
            w_update[j] = np.dot(pulls, X[j]) #refer back to the equation, delta w_j = Sigma {(y_hat - y) x_j}
            
        w -= (alpha/len(x)) * w_update
        b -= (alpha/len(x)) * np.sum(pulls)
        cost.append( np.sum( pulls**2 ) / (2*len(x)) )
        
    w, b = scale_params(X_unscaled, w, b, 'undo')
    return w, b, cost



# %%

x, y = training_set_gen(func1, -1, 1, 100)
plt.plot(x, y, 'b.')
plt.plot(x, func1(x), 'k-')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('Training Set')
plt.show()

#%%

order = 4
learning_rate, n_iteration = 1e-2, 10000
w, b, cost = polynomial_regression(x, y, order, learning_rate, n_iteration)

plt.plot(cost)
plt.xlabel('Number of Iterations')
plt.ylabel('Cost')
plt.show()

# %%

polyfit_params = np.polyfit(x, y, order)
regression_params = np.append(w[::-1], b)

print(f'NumPy polyfit params: {polyfit_params}')
print(f'polynomial rg params: {regression_params}')

y_hat1, y_hat2 = np.polyval(regression_params, feature_translation(x)), np.polyval(polyfit_params, x)
plt.plot(x, y, 'k.', label = 'Training Set')
plt.plot(x, y_hat1, 'g-', label = 'regression')
plt.plot(x, y_hat2, 'r-', label = 'polyfit')
plt.legend()
plt.show()
    


