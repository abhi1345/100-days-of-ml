# Learning Vector Quantization for 2-class Classification

dataset = {
(3, 2) : 0,
(0, 2) : 0,
(1, 2) : 0,
(0, 0) : 0,
(4, 2) : 0,
(5, 6) : 1,
(7, 6) : 1,
(9, 9) : 1,
(13, 11) : 1,
(13, 12) : 1
}

class LVQ:
    def __init__(self):
        self.var = True

    def train(self, dataset):
        self.data = dataset
        self.classes = set(dataset.values())
        self.oneEpoch(15)

    def oneEpoch(self, epochs):
        self.vectors()
        for e in range(epochs):
            self.alpha = 0.3 * (1-(e+1)/epochs)
            self.onePattern()


    def onePattern(self):

        self.alpha = 0.3
        for k,v in self.data.items():
            mind, bmu, index = float('inf'), 0, 0
            i = 0
            for vec, output in self.codebook.items():
                currdist = self.euclid(vec, k)
                if currdist < mind:
                    mind, bmu, index = currdist, vec, i
                i += 1

            oldbmu = bmu
            oldout = self.codebook[bmu]
            c = 1 if v == self.codebook[bmu] else -1
            bmu = list(bmu)
            bmu[0] += c*self.alpha*(k[0]-bmu[0])
            bmu[1] += c*self.alpha*(k[1]-bmu[1])
            bmu = tuple(bmu)

            del self.codebook[oldbmu]
            self.codebook[bmu] = oldout


    def vectors(self):
        self.codebook = {}
        for c in self.classes:
            initvectors = self.keyfromd(c, 2)
            self.codebook.update(initvectors)


    def predict(self, p):
        mind, bmu, index = float('inf'), 0, 0
        i = 0
        for vec, output in self.codebook.items():
            currdist = self.euclid(vec, p)
            if currdist < mind:
                mind, bmu, index = currdist, vec, i
            i += 1

        return self.codebook[bmu]



    # HELPER FUNCTIONS
    #Return key with corresponding value

    def keyfromd(self, v, amt):
        ret, size = {}, 0
        for key,val in self.data.items():
            if val == v:
                ret[key] = val
                size += 1
            if size >= amt:
                return ret

    def euclid(self, v1, v2):
        d = 0
        for i in range(len(v1)):
            d += (v1[i]-v2[i])**2
        return d**0.5

model = LVQ()
model.train(dataset)
t1, t2 = (0,2), (7,9)
print(model.predict(t1), model.predict(t2)) #Should be 0, 1
