# k Nearest Neighbors Algorithm

dataset = {(1,2):4, (2,5):6, (5,7):10, (1,5):4, (3,5):8}
dataset2 = {(1,2):1, (2,5):0, (5,7):0, (1,5):1, (3,5):1}

class kNearestNeighbor:

    def __init__(self, dataset, k, regBool):
        self.data = dataset
        self.k = k #Must be less than len(dataset)
        self.n = {}
        self.regression = regBool

    def update(self, p, d):
        if len(self.n) < self.k:
            self.n[p] = d
            return

        for key in self.n.keys():
            if d < self.n[key]:
                self.n[p] = d
                break

        if len(self.n) > self.k:
            obsolete = max(self.n, key=self.n.get)
            del self.n[obsolete]

    def dist(self, p1, p2):
        """Euclidian Distance"""
        xd = (p1[0] - p2[0])**2
        yd = (p1[1] - p2[1])**2
        return (xd + yd)**0.5

    def avg(self, l):
        """Helper Function (mean)"""
        l = list(l)
        return sum(l) / len(l)

    def mode(self, l):
        l = list(l)
        l.sort()
        n, c = l[0], 1
        tempc = 1
        for i in range(1, len(l)):
            if l[i] == l[i-1]:
                tempc += 1
            else:
                tempc = 1
            if tempc > c:
                c = tempc
                n = l[i]
        return n


    def regress(self, inp):
        return self.avg([self.data[x] for x in self.n.keys()])

    def classify(self, inp):
        return self.mode([self.data[x] for x in self.n.keys()])

    def predict(self, inp):
        """Resets neighborlist, allowing multiple predictions with 1 model."""
        self.n = {}
        for p in dataset.keys():
            self.update(p, self.dist(p, inp))

        if self.regression:
            return self.regress(inp)
        else:
            return self.classify(inp)

    def train(self, dataset):
        return 0

# Uncomment the Below Lines to Test kNN Implementation
"""
model = kNearestNeighbor(dataset, 2, True)
out1 = model.predict((1,2))
model = kNearestNeighbor(dataset2, 3, False)
out2 = model.predict((3,4))
print(out1, out2)
"""
