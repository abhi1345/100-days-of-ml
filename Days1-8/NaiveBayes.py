#Simple Naive Bayes in Python, using Bag of Words model for Spam Classification
#Laplace Smoothing Applied when Calculating likelihoods

dataset = {
"Get your free coupon today!" : "spam",
"Hi John. Let's have lunch this weekend." : "ham",
"Don't miss out on our once in a lifetime opportunity!" : "spam"
}


class NaiveBayes:

    def __init__(self):
        self.made = True

    def train(self, dataset):
        self.data = dataset
        self.classProbs = {}
        self.likelihoods = {}
        self.totals = {}
        self.bag = set()

        self.classProbabilities()
        self.bagbuilder()
        self.wordtotals()
        self.calclikelihoods()

    def predict(self, text):
        text = self.clean(text)
        text = text.split(" ")

        p = 1

        currp, currcat = 0, ""

        for cat in self.classProbs.keys():
            p = self.classProbs[cat]
            for word in text:
                t = (word, cat)
                p *= self.likelihoods[t]
            if p > currp:
                currp, currcat = p, cat

        return currcat


    #Training Helper Functions.
    def classProbabilities(self):

        for v in self.data.values():
            if v in self.classProbs:
                self.classProbs[v] += 1
            else:
                self.classProbs[v] = 1

        for k in self.classProbs:
            self.classProbs[k] /= len(self.data)

    def bagbuilder(self):
        for text in self.data.keys():
            for word in text.split(" "):
                word = self.clean(word)
                for cat in self.data.values():
                    self.likelihoods[(word, cat)] = 0
                v = self.data[text]
                self.update(self.likelihoods, (word, v), 1)
                self.bag.add(word)

    def wordtotals(self):
        for k in self.data.keys():
            v = self.data[k]
            if v in self.totals:
                self.totals[v] += len(k.split(" "))
            else:
                self.totals[v] = len(k.split(" "))

    def calclikelihoods(self):
        for k in self.likelihoods:
            cat = k[1]
            self.likelihoods[k] = (self.likelihoods[k]+1) / (self.totals[cat]+1)

    # Helper Functions, one to update a dictionary, one to clean a string of punctuation.
    def update(self, d, k, v):
        if k in d.keys():
            d[k] += v
        else:
            d[k] = v
        return d

    def clean(self, s):
        for c in ['.',',','?','!']:
            s = s.replace(c, "")
        return s



model = NaiveBayes()
model.train(dataset)
print(model.predict("John"))
