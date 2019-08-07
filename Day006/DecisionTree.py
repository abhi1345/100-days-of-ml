# Decision Trees for 2-class classification
# Cost function = Gini Index

dataset = {
(2.771244718, 1.784783929) : 0,
(1.728571309, 1.169761413) : 0,
(3.678319846, 2.81281357) : 0,
(3.961043357, 2.61995032) : 0,
(2.999208922, 2.209014212) : 0,
(7.497545867, 3.162953546) : 1,
(9.00220326, 3.339047188) : 1,
(7.444542326, 0.476683375) : 1,
(10.12493903, 3.234550982) : 1,
(6.642287351, 3.319983761) : 1
}

class DecisionTree:

    def __init__(self):
        self.made = True

    def train(self, data):
        self.data = data
        self.classes = set(self.data.values())


        optimalg, optimalpair, optimalindex = 2, 0, 0

        for p in self.data.keys():
            t = self.split(p)
            gini, i = t

            if gini < optimalg:
                optimalg, optimalpair, optimalindex = gini, p, i


        self.split = optimalpair[i]
        self.pairindex = optimalindex




    def gini(self, d):
        a = 0
        for k in d.keys():
            v = d[k]
            a += v*(1-v)
        return a


    def split(self, p):
        index = 0
        maxs, maxgini = 0, 2
        for split in p:

            counts = {}
            for clss in self.classes:
                t1, t2 = (clss, 'left'), (clss, 'right')
                counts[t1] = 0
                counts[t2] = 0

            left, right = 0, 0
            for pair in self.data.keys():
                v = self.data[pair]
                if pair[index] >= split:
                    side = 'right'
                    right += 1
                else:
                    side = 'left'
                    left += 1
                tup = (v, side)
                counts[tup] += 1


            for k in counts.keys():
                try:
                    counts[k] = counts[k]/right if k[1] == 'right' else counts[k]/left
                except ZeroDivisionError:
                    counts[k] = 0

            tempgini = self.gini(counts)

            if tempgini < maxgini:
                maxgini = tempgini
                maxs = index


            index += 1

        return (maxgini, maxs)



    def predict(self, p):
        comp = p[self.pairindex]

        if comp >= self.split:
            return 1
        else:
            return 0




model = DecisionTree()
model.train(dataset)

validationSet = {
(2.343875381, 2.051757824) : 0,
(3.536904049 ,3.032932531) : 0,
(2.801395588 ,2.786327755) : 0,
(3.656342926, 2.581460765) : 0,
(2.853194386, 1.052331062) : 0,
(8.907647835, 3.730540859) : 1,
(9.752464513, 3.740754624) : 1,
(8.016361622, 3.013408249) : 1,
(6.68490395, 2.436333477) : 1,
(7.142525173, 3.650120799) : 1
}

#Validation: Accuracy should be 100%
accuracy = 0
divisor = 0
for k in validationSet:
    v = validationSet[k]
    if model.predict(k) == v:
        accuracy += 1
    divisor += 1

print("Model accuracy: {}%".format(accuracy/divisor*100))
