import numpy as np
import pandas as pd
from sklearn import linear_model

""" def gradient_descent(x,y):
    current_a = 0  # Sometimes known as m
    current_b = 0
    learning_rate = 0.0001  # alpha
    n = len(x)
    iteration = 10
    for i in range(iteration):
        y_predicted = current_a * x + current_b
        cost = (1 / n) * sum([val ** 2 for val in (y - y_predicted)])

        derivative_a = -(2/n) * sum(x*(y - y_predicted))
        derivative_b = -(2/n) * sum(y - y_predicted)

        current_a = current_a - learning_rate * derivative_a
        current_b = current_b - learning_rate * derivative_b

        print("a = {}, b = {}, iteration = {}, cost = {}". format(current_a,current_b,i,cost)) """


x = np.array([[30,3,6], [43,4,8], [25,2,3], [51,4,9], [40,3,5], [20,1,2]])
y = np.array([2.5,3.4,1.8,4.5,3.2,1.6])

theta = np.array([])
print(x[1])