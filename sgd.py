import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class logisticRegression():
    def __init__(self, lr, d, reg=0.01):
        self.w = np.zeros(d)
        self.lr = lr
        self.reg = reg
    
    def train(self, rdd, iteration=10000):
        N = rdd.count()
        it = 0
        # loss = rdd.map(lambda point: np.log(1 + np.exp(-point.label * np.dot(point.features, self.w)))).sum() / N
        while it < iteration:
            self.w += self.lr / N * rdd.map(lambda point: sigmoid(-point.label * np.dot(point.features, self.w)) * \
                                            (point.label * point.features)).reduce(lambda r1, r2: r1 + r2) - 2 * self.reg * self.w
            it += 1
    
    def predict(self, x):
        pred = sigmoid(np.dot(x.T, self.w))
        return 1 if pred > 0.5 else 0