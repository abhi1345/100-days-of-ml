#Linear Discriminant Analysis
from math import log
def mean(l):
    return sum(l)/len(l)


class LDA:

    def __init__(self):
        self.size = 0
        self.data = {}
        self.probs = {}
        self.means = {}

    def train(self, dataset):
        self.data = dataset
        self.size = len(dataset)
        self.calcprobs()
        self.numclasses = len(self.probs)
        self.xmean = mean(list(self.probs.values())) * self.size
        self.calcmeans()
        self.calcvars()

    def calcmeans(self):
        s = {}
        i = 0
        for k in self.data.keys():
            v = self.data[k]
            i +=  1
            try:
                s[v] += k
            except KeyError:
                s[v] = k
        for k in s.keys():
            s[k] /= i*self.probs[k]
        self.means = s

    def calcprobs(self):
        s = {}
        i = 0
        for v in self.data.values():
            i +=  1
            try:
                s[v] += 1
            except KeyError:
                s[v] = 1
        for k in s.keys():
            s[k] /= i
        self.probs = s

    def calcvars(self):
        self.sqdiffs = {}
        for c in self.probs:
            s = 0
            for k in self.data:
                if self.data[k] == c:
                    s += self.sqdiff(k)

            self.sqdiffs[c] = s


        xcount = self.probs[c]*self.size
        classcount = len(self.probs)
        vc = 1/(self.xmean-self.numclasses)
        v = sum(list(self.sqdiffs.values())) * vc

        self.variance = v



    def sqdiff(self, x):
        m = self.means[self.data[x]]
        return (x-m)**2



    def predict(self, x):
        answer, m = 0, float('-inf')
        for c in self.probs.keys():
            d = self.discriminant(x, c)
            if d > m:
                m = d
                answer = c

        return answer

    def discriminant(self, x, y):
        f1 = self.means[y] / self.variance
        f2 = self.means[y]**2 / (2*self.variance)
        logprob = log(self.probs[y])
        return x*f1-f2+logprob


dataset = {
4.667797637 : 0,
5.509198779 : 0,
4.702791608 : 0,
5.956706641 : 0,
5.738622413 : 0,
5.027283325 : 0,
4.805434058 : 0,
4.425689143 : 0,
5.009368635 : 0,
5.116718815 : 0,
6.370917709 : 0,
2.895041947 : 0,
4.666842365 : 0,
5.602154638 : 0,
4.902797978 : 0,
5.032652964 : 0,
4.083972925 : 0,
4.875524106 : 0,
4.732801047 : 0,
5.385993407 : 0,
20.74393514 : 1,
21.41752855 : 1,
20.57924186 : 1,
20.7386947 : 1,
19.44605384 : 1,
18.36360265 : 1,
19.90363232 : 1,
19.10870851 : 1,
18.18787593 : 1,
19.71767611 : 1,
19.09629027 : 1,
20.52741312 : 1,
20.63205608 : 1,
19.86218119 : 1,
21.34670569 : 1,
20.333906 : 1,
21.02714855 : 1,
18.27536089 : 1,
21.77371156 : 1,
20.65953546 : 1, }

model = LDA()
model.train(dataset)
print(model.predict(24.6677)) #should be 1
print(model.predict(3)) #should be 0
