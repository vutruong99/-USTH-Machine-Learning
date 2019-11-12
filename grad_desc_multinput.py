import numpy as np
import matplotlib.pyplot as plt


def cost_f(x, y, theta):
    n = len(x)
    predictions = x.dot(theta)
    cost = (1 / (2*n)) * np.sum(np.square(predictions - y))
    return cost


def stochastic(x, y):
    theta = np.array([0, 0, 0, 0])
    theta.shape += (1,)
    n = len(x)
    learning_rate = 0.00001  # alpha
    iteration = 100

    costs = np.zeros(iteration)  # Table of cost values after each iteration
    thetas = np.zeros((iteration, 4))  # Table of theta values after each iteration

    for i in range(iteration):
        for j in range(n):
            random_index = np.random.randint(0,n)

            x_r = x[random_index, :].reshape(1, x.shape[1])
            y_r = y[random_index, :].reshape(1, 1)

            prediction = np.dot(x_r, theta)
            theta = theta - (1 / n) * learning_rate * (x_r.T.dot((prediction - y_r)))
            thetas[i, :] = theta.T
            cost = cost_f(x,y,theta)
        costs[i] = cost

    print("STOCHASTIC GRADIENT DESCENT")
    print("theta0 = {:0.5f}, theta1 = {:0.5f}, theta2 = {:0.5f}, theta3 = {:0.5f}".format(theta[0][-1], theta[1][-1],
                                                                                          theta[2][-1], theta[3][-1]))
    print("cost = {:0.5f}".format(costs[-1]))
    print("")
    return costs


def mini_batch(x, y):
    batch_size = 2
    theta = np.array([0, 0, 0, 0])
    theta.shape += (1,)
    n = len(x)
    learning_rate = 0.00001  # alpha
    iteration = 100

    costs = np.zeros(iteration)  # Table of cost values after each iteration
    thetas = np.zeros((iteration, 4))  # Table of theta values after each iteration

    for i in range(iteration):
        indexes = np.random.permutation(n)
        x = x[indexes]
        y = y[indexes]
        for j in range(0, n, batch_size):
            x_j = x[j:j+batch_size]
            x_j = np.c_[np.ones(len(x_j)), x_j]  # Add one to the X matrix
            y_j = y[j:j+batch_size]
            prediction = np.dot(x_j, theta)
            theta = theta - (1 / batch_size) * learning_rate * (x_j.T.dot((prediction - y_j)))
            thetas[i, :] = theta.T

        costs[i] = cost_f(np.c_[np.ones(len(x)),x],y,theta)

    print("MINI BATCH GRADIENT DESCENT")
    print("theta0 = {:0.5f}, theta1 = {:0.5f}, theta2 = {:0.5f}, theta3 = {:0.5f}".format(theta[0][-1], theta[1][-1], theta[2][-1], theta[3][-1]))
    print("cost = {:0.5f}".format(costs[-1]))
    print("")

    return costs


def gradient_descent(x, y):
    theta = np.array([0, 0, 0, 0])
    theta.shape += (1,)
    n = len(x)
    learning_rate = 0.00001  # alpha
    iteration = 100

    costs = np.zeros(iteration)  # Table of cost values after each iteration
    thetas = np.zeros((iteration, 4))  # Table of theta values after each iteration

    for i in range(iteration):
        prediction = np.dot(x, theta)
        theta = theta - (1 / n) * learning_rate * (x.T.dot((prediction - y)))
        thetas[i, :] = theta.T
        costs[i] = cost_f(x, y, theta)

    print("BATCH GRADIENT DESCENT")
    print("theta0 = {:0.5f}, theta1 = {:0.5f}, theta2 = {:0.5f}, theta3 = {:0.5f}".format(theta[0][-1], theta[1][-1], theta[2][-1], theta[3][-1]))
    print("cost = {:0.5f}".format(costs[-1]))
    print("")


    return costs


def normal_equation(x , y):
    print("NORMAL EQUATION THETAS")
    print(np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y))


x = np.array([[30, 3, 6], [43, 4, 8], [25, 2, 3], [51, 4, 9], [40, 3, 5], [20, 1, 2]])
x_oneAdded = np.c_[np.ones((len(x), 1)), x]

y = np.array([2.5, 3.4, 1.8, 4.5, 3.2, 1.6])
y.shape += (1,)

costg = gradient_descent(x_oneAdded,y)
costm = mini_batch(x,y)
costs = stochastic(x_oneAdded,y)
normal_equation(x_oneAdded,y)

plt.xlabel("Epochs")
plt.ylabel("Costs")
plt.title("Costs after 100 epochs")
plt.plot(np.linspace(1,100,num=100),costg,'b',label="Batch")
plt.plot(np.linspace(1,100,num=100),costm,'g',label="Mini-batch")
plt.plot(np.linspace(1,100,num=100),costs,'r',label="Stochastic")
plt.legend()
plt.show()










