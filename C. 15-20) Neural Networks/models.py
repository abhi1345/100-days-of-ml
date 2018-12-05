import nn

#All Functions were filled in by me
#Skeleton code provided by Berkeley's CS 188 Course Staff

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        answer = nn.DotProduct(self.w, x)
        return answer
        "*** YOUR CODE HERE ***"

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        dp = nn.as_scalar(self.run(x))
        answer = 1 if dp >= 0 else -1
        return answer
        "*** YOUR CODE HERE ***"

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        #print(dataset.x, dataset.y)
        conv = False
        count = 0
        while not conv:
            conv = True
            for x, y in dataset.iterate_once(1):
                yhat, ystar = self.get_prediction(x), nn.as_scalar(y)
                if yhat != ystar:
                    conv = False
                    self.w.update(x, ystar)


        "*** YOUR CODE HERE ***"

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        hidden_size = 200


        self.w1 = nn.Parameter(1, hidden_size)
        self.w2 = nn.Parameter(hidden_size, 1)
        self.b1 = nn.Parameter(1, hidden_size)
        self.b2 = nn.Parameter(1, 1)

        self.lr = 0.08


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"

        dp = nn.Linear(x, self.w1)
        #batch_size x 1 * 1*100
        #output = 1x100
        #print("DP done")
        bsum = nn.AddBias(dp, self.b1)
        #output = 1x100
        #print("BSUM Done")
        t1 = nn.ReLU(bsum)
        #print("RELU Done")
        dp2 = nn.Linear(t1, self.w2)
        #print("DP2 done")
        return nn.AddBias(dp2, self.b2)


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"

        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        batch_size = len(dataset.x)//5
        for epoch in range(2000):
            for x, y in dataset.iterate_once(batch_size):
                #yhat = self.run(x)
                loss = self.get_loss(x, y)
                params = [self.w1, self.b1, self.w2, self.b2]
                grads = nn.gradients(loss, params)

                for i,p in enumerate(params):
                    p.update(grads[i], -self.lr)





        "*** YOUR CODE HERE ***"

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.hidden_size = 300


        self.w1 = nn.Parameter(784, self.hidden_size)
        self.w2 = nn.Parameter(self.hidden_size, 10)
        self.b1 = nn.Parameter(1, self.hidden_size)
        self.b2 = nn.Parameter(1, 10)

        self.l3 = nn.Parameter(10, 10)
        self.b3 = nn.Parameter(1, 10)

        self.lr = 0.1


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        #x: batchx784, w1: 784xhiddensize

        dp = nn.Linear(x, self.w1)
        #output = batch x hidden_size
        #print("DP done")
        bsum = nn.AddBias(dp, self.b1)
        #output = batch x hidden_size
        #print("BSUM Done")
        t1 = nn.ReLU(bsum)
        #print("RELU Done")
        #output = batch x hidden_size

        dp2 = nn.Linear(t1, self.w2)
        #print("DP2 done")
        # output = batch x 10

        return nn.AddBias(dp2, self.b2)


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        batch_size = 6000//10
        for epoch in range(50):
            for x, y in dataset.iterate_once(batch_size):
                #yhat = self.run(x)
                loss = self.get_loss(x, y)
                params = [self.w1, self.b1, self.w2, self.b2]
                grads = nn.gradients(loss, params)

                for i,p in enumerate(params):
                    p.update(grads[i], -self.lr)


class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

        #Hyperparameters - CHANGE THESE
        self.batch_size = 200
        self.learning_rate = .04

        hidden_layers = 400

        self.w0 = nn.Parameter(self.num_chars, hidden_layers)
        self.w1 = nn.Parameter(self.num_chars, hidden_layers)
        self.w2 = nn.Parameter(hidden_layers, hidden_layers)
        self.b0 = nn.Parameter(1, hidden_layers)
        self.b1 = nn.Parameter(1, hidden_layers)
        self.final_weight = nn.Parameter(hidden_layers, 5)
        self.final_bias = nn.Parameter(1, 5)


    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"


        #Side Comments list dimensions of matrices for reference.
        first = xs[0] #batch x numchars
        prod = nn.Linear(first, self.w0) #batch x hidden_size
        vsum = nn.AddBias(prod, self.b0) #batch x hidden_size
        x = nn.ReLU(vsum) #batch x hidden_size

        for node in xs[1:]: #batch x numchars
            fac1 = nn.Linear(node, self.w1) #batch x hidden
            fac2 = nn.Linear(x, self.w2) #batch x hidden_layers
            vsum = nn.Add(fac1, fac2) #batch x hidden_layers
            x = nn.AddBias(vsum, self.b1) #batch x hidden_layers

        x = nn.ReLU(x)
        x = nn.Linear(x, self.final_weight) #batch x 5
        x = nn.AddBias(x, self.final_bias) #batch x 5

        answer = x

        return answer

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(xs), y)

    def train(self, dataset):
        """
        Trains the model.
        Uses gradient descent.
        """
        "*** YOUR CODE HERE ***"
        epochs = 50
        for e in range(epochs):
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)

                p = [self.w0, self.w1, self.w2, self.final_weight, self.b0, self.b1, self.final_bias]

                grads = nn.gradients(loss, p)

                for i,p in enumerate(p):
                    p.update(grads[i], -self.learning_rate)
