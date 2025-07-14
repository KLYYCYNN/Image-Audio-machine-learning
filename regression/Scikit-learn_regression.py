
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model as lm

# %%

np.set_printoptions(precision=4)

def func1(x): # any function for training data generation
    return np.cos(5*x) + np.sin(3*x) - np.exp(-x**2/11)

def func2(x):
    return 1e3 * np.exp( np.cos(x)**2 ) / (x**2 + 3*x + 1)

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

def skl_polyfit(x, y, deg):
    power = (np.arange(deg)+1)[::-1]
    
    X = np.zeros((len(x), deg))
    for i in range(deg):
        X[:,i] = x ** power[i]
        
    fit = lm.LinearRegression().fit(X,y)
    
    return np.append( fit.coef_, fit.intercept_ )
    
    

# %%

x, y = training_set_gen(func2, 30, 35, 100)
plt.plot(x, y, 'b.')
plt.plot(x, func2(x), 'k-')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('Training Set')
plt.show()

# %%
degree = 6
params1, params2 = skl_polyfit(x, y, degree), np.polyfit(x, y, degree)
print(f'np:  {params2}')
print(f'skl: {params1}')
plt.plot(x, y, 'k.')
plt.plot(x, np.polyval(params1, x), 'r-', label = 'skl')
plt.plot(x, np.polyval(params2, x), 'g-', label = 'np')
plt.legend()
plt.show()
