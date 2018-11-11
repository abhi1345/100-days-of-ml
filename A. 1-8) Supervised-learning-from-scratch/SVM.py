#2-Class Classificatin with Support Vector Machine
#Trained with sub-gradient descent (random training pattern)

dataset = {
(2.327868056, 2.458016525) : -1,
(3.032830419, 3.170770366) : -1,
(4.485465382, 3.696728111) : -1,
(3.684815246, 3.846846973) : -1,
(2.283558563, 1.853215997) : -1,
(7.807521179, 3.290132136) : 1,
(6.132998136, 2.140563087) : 1,
(7.514829366, 2.107056961) : 1,
(5.502385039, 1.404002608) : 1,
(7.432932365, 4.236232628) : 1
}

class SupportVectorMachine:
    def __init__(self):
        self.b0 = 0 #Deprecated for simple case, represents y-intercept
        self.b1 = 0
        self.b2 = 0
        self.lam = 0.0001

    def train(self, data, epochs):
        t = 0
        for e in range(epochs):
            t += 1
            for pair in data.keys():
                x1, x2, y = pair[0], pair[1], data[pair]
                output = self.test(y, x1, x2)
                if output > 1:
                    self.update1(t)
                else:
                    self.update2(t, x1, x2, y)

    def predict(self, p):
        x1, x2 = p[0], p[1]
        output = self.b1*x1 + self.b2*x2
        answer = -1 if output < 0 else 1
        return answer


    def test(self, y, x1, x2):
        return y*self.b1*x1 + self.b2*x2

    def update1(self, t):
        self.b1 *= 1-(1/t)
        self.b2 *= 1-(1/t)

    def update2(self, t, x1, x2, y):
        self.b1 = (1-(1/t))*self.b1 + (y*x1)/(self.lam*t)
        self.b2 = (1-(1/t))*self.b2 + (y*x2)/(self.lam*t)




model = SupportVectorMachine()
model.train(dataset, 10)
tup = (6,2)

testdata = {
(1, 4) : -1,
(3, 4) : -1,
(1, 1) : -1,
(9, 1) : 1,
(8, 4) : 1
}

#Accuracy Check
acc, div = 0, 0
for t in testdata.keys():
    if model.predict(t) == testdata[t]:
        acc += 1

    div += 1

