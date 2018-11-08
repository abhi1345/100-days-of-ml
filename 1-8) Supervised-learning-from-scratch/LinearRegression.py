# Stochastic Gradient Descent for Linear Regression
#You can modify below datasets to model different linear equations.

dataset = {1:2, 2:4, 3:6, 7:14}
dataset2 = {1:3, 2:5, 6:13, 10:21}

class LinearRegression:

  def __init__(self, data):
    self.dataset = data
    self.a = 0
    self.b = 0

  def real(self, x):
    return self.dataset[x]

  def linear(self, a, b, x):
    return a*x + b

  def error(self, a, b, i, yhat=None):
    if yhat is None:
      return self.linear(a, b, i) - self.real(i)
    else:
      return self.linear(a, b, i) - yhat

  def gradientDescent(self, alpha, epochs):

    a, b = 0, 0

    for i in range(epochs):
      for x in self.dataset.keys():
        y = self.dataset[x]
        e = self.error(a, b, x, y)
        a = a - (alpha*e*x)
        b = b - (alpha*e)

    
    self.a, self.b = a, b
    return (a, b)

  def train(self):
    return self.gradientDescent(0.01, 10000)

  def predict(self, x):
    return self.linear(self.a, self.b, x)


#Uncomment the below lines for a demonstration of linear regression
"""
model = LinearRegression(dataset)
function = model.train()
print("Model parameters: {}, {}".format(function[0], function[1]))
print("Model equation: y = {}x + {}".format(function[0], function[1]))
print("Prediction for {} is {}".format(5, model.predict(23)))
"""

