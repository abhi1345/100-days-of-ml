# Stochastic Gradient Descent for Linear Regression
#You can modify below datasets to model different linear equations.

class LinearRegression:


    def __init__(self):
        self.a = 0
        self.b = 0


    def linear(self, a, b, x):
        return a*x + b


    def error(self, a, b, i, data, yhat=None):
        if yhat is None:
            return self.linear(a, b, i) - data[i]
        else:
            return self.linear(a, b, i) - yhat

    def gradientDescent(self, data, alpha, epochs):
        a, b = 0, 0

        for i in range(epochs):
            for x in data.keys():
                y = data[x]
                e = self.error(a, b, x, data, y)
                a = a - (alpha*e*x)
                b = b - (alpha*e)

        self.a, self.b = a, b

        return (a, b)

    def train(self, data):
        return self.gradientDescent(data, 0.001, 100000)

    def predict(self, x):
        return self.linear(self.a, self.b, x)
