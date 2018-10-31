#Logistic Regression with Stochastic Gradient Descent

dataset = {
(2.7810836, 2.550537003) : 0,
(1.465489372, 2.362125076) : 0,
(3.396561688, 4.400293529) : 0,
(1.38807019, 1.850220317) : 0,
(3.06407232, 3.005305973) : 0,
(7.627531214, 2.759262235) : 1,
(5.332441248, 2.088626775) : 1,
(6.922596716, 1.77106367) : 1,
(8.675418651, -0.242068655) : 1,
(7.673756466, 3.508563011) : 1
}

class LogisticRegression:

    def __init__(self):
        self.w1, self.w2, self.w3 = 0, 0, 0

    def linear(self, x1, x2):
        return x1*self.w1 + x2*self.w2 + self.w3

    def sigmoid(self, v):
        e = 2.71828
        return 1/(1 + e**(-v))

    def pdef(self, x1, x2):
        return self.sigmoid(self.linear(x1, x2))

    def train(self, dataset, alpha, epochs):
        self.data = dataset

        for e in range(epochs):
            for pair in self.data.keys():
                x1, x2 = pair
                y = self.data[pair]
                prediction = self.pdef(x1, x2)
                self.w1 += alpha*(y-prediction)*prediction*(1-prediction)*x1
                self.w2 += alpha*(y-prediction)*prediction*(1-prediction)*x2
                self.w3 += alpha*(y-prediction)*prediction*(1-prediction)

    def predict(self, inputpair):
        x, y = inputpair
        return self.pdef(x, y)

    def classify(self, inputpair):
        p = self.predict(inputpair)
        answer = 1 if p >= 0.5 else 0
        return answer

    def accuracy(self, testset):
        accuracy = 0
        for pair in testset.keys():
            y = testset[pair]
            x1, x2 = pair
            p = model.classify(pair)
            if p == y:
                accuracy += 1
        return accuracy / len(testset)



model = LogisticRegression()
model.train(dataset, 0.3, 10)
print(model.accuracy(dataset)) #Should be 1, since we trained on this set
