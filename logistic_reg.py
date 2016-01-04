import numpy as np
import matplotlib.pyplot as plt
import random


class Logistic_reg:
    def __init__(self):
        self.w = np.zeros((3, 1))
        self.data = None
        self.target = None
        self.func = None

    def gen_targetfunc(self, N):
        # Get the target function
        w = np.array([1, np.random.uniform(-.5, .5),
                      np.random.uniform(-5, .5)])[np.newaxis]
        # print w.T

        # Generate training data from it
        X = np.random.uniform(-1, 1, (N, 2))
        X = np.hstack((np.ones((N, 1)), X))

        self.data = X
        self.func = w
        y = np.inner(w, X)
        self.target = y
        self.target[y > 0] = 1
        self.target[y < 0] = -1
        # print self.target

    def display(self, X, y, w):
        # create a mesh to plot in
        h = .02
        x_min, x_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        y_min, y_max = X[:, 2].min() - 1, X[:, 2].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        Z = np.inner(w, np.c_[np.ones(xx.ravel().shape),
                              xx.ravel(),
                              yy.ravel()])
        Z[Z >= 0] = 1
        Z[Z < 0] = -1
        Z = Z.reshape(xx.shape)
        # print 'Z', Z
        plt.figure()
        plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
        # Plot also the training points
        plt.scatter(X[:, 1], X[:, 2], c=y, cmap=plt.cm.Paired)
        plt.xlabel('Sepal length')
        plt.ylabel('Sepal width')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.show()

    def logistic_regression(self):
        if self.data is None:
            pass
        idx = random.sample(range(0, 100), 100)
        while idx:
            sample_id = idx.pop()
            print self.data[sample_id, :]
            



logistic = Logistic_reg()
logistic.gen_targetfunc(100)
logistic.display(logistic.data, logistic.target, logistic.func)
logistic.logistic_regression()
