
import numpy as np
import matplotlib.pyplot as plt

# %%

np.set_printoptions(precision=8)

# %%

def line(a, b, x):
    return a*x + b

#Creates a random array of 1 and -1
def pm_arr(n):
    pm = [-1, 1]
    pm_ind = np.random.randint(0, 2, n)
    pm_arr = np.array( [pm[i] for i in pm_ind] )
    return pm_arr

def training_set_gen(a, b, xmin, xmax, n):
    x = np.linspace(xmin, xmax, 5*n)
    del_ind = np.random.choice(5*n, 4*n, replace = False) # creates 5n x-values and randomly deletes 4n of them, leaving only n
    x = np.delete(x, del_ind)
    
    y = line(a, b, x)
    d = max(y) - min(y)
    y_flu = d * 0.2 * np.random.rand(n) * pm_arr(n) #random fluctuations, with maximum amplitude d/5
    y += y_flu
    
    return x, y

def linear_regression(x, y, alpha, n):
    #inital guesses
    a = (y[-1] - y[0]) / (x[-1] - x[0])
    rand_ind = np.random.randint(0, len(x))
    b = y[rand_ind] - a*x[rand_ind]
    m = len(x)
    #gradient descent
    for i in range(n):
        a_update_step = (alpha/m) * np.sum( (line(a, b, x) - y) * x )
        b_update_step = (alpha/m) * np.sum( (line(a, b, x) - y) )
        
        a -= a_update_step
        b -= b_update_step
        
    return a, b
        
# %%

a, b = 3, 5 #set parameters
x, y = training_set_gen(a, b, 0, 10, 100) # generate training set

plt.plot(x, y, 'r.')
plt.plot(x, line(a, b, x), 'k-')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('Training Set')
plt.show()

a_reg, b_reg = linear_regression(x, y, 0.01, 10000) #learning rate 0.01, 10000 iterations
print(f'linear regression params: {np.array([a_reg, b_reg])}')
print(f'NumPy polyfit params: {np.polyfit(x, y, 1)}')
    
    
    
    
    


