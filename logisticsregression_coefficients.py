import math

input = [2.5,3.5,5.6,2.2,6.9,9.6]
expectedOutput = [0,0,1,1,0,1]
alpha = 0.0001
theta = [0,0]
theta0 = 0
theta1 = 0



def hOfX(x):
    return 1/(1 + math.exp(theta[0] + theta[1]*x))

def derivateiveTheta(j):
    cost_sum = 0
    if j==0:
        for i in range(len(input)):
            cost_sum = cost_sum + (hOfX(input[i]) - expectedOutput[i])
    else: 
        for i in range(len(input)):
            cost_sum = cost_sum + (hOfX(input[i]) - expectedOutput[i])*input[i]

    return (1/len(input))*cost_sum

def newTheta(j):
    newTheta = theta[j] - alpha * derivateiveTheta(j)
    return newTheta

print(newTheta(0))
print(newTheta(1))
