# k Nearest Neighbors Algorithm

class kNearestNeighbor:

    def __init__(self, dataset, k, regBool):
        self.k = k #Must be less than len(dataset)
        self.n = {}
        self.regression = regBool
        self.data = dataset

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
        for p in self.data.keys():
            self.update(p, self.dist(p, inp))

        if self.regression:
            return self.regress(inp)
        else:
            return self.classify(inp)

