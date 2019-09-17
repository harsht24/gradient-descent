import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv('Salary_Data.csv')

x=dataset['YearsExperience']
y=dataset['Salary']

# plt.scatter(x,y)
# plt.show()

slope = 0
intercept = 0
learning_rate = 0.01
epochs = 1000

n = float(len(x))


for i in range(epochs):
    Y_pred = slope * x + intercept
    derivative_slope = (-2 / n) * sum(x * (y - Y_pred))
    derivative_intercept = (-2 / n) * sum(y - Y_pred)
    slope = slope - learning_rate * derivative_slope
    intercept = intercept - learning_rate * derivative_intercept

print("Value of slope is = ",intercept)
print("Value of intercept is = ",slope)
Y_pred = slope * x + intercept


mean_squared_error = sum((y - Y_pred) ** 2) / n
root_mean_squared_error = np.sqrt(mean_squared_error)
print("Root mean squared error: ", root_mean_squared_error)

ssr = sum((y - Y_pred) ** 2)
sst = sum((y - np.mean(y)) ** 2)
r2_score = 1 - (ssr / sst)
print("R2 score or coefficient of determination: ", r2_score)

plt.scatter(x, y)
plt.plot([min(x), max(x)], [min(Y_pred), max(Y_pred)], color='blue') # predicted
plt.show()

