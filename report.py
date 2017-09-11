import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.datasets import load_iris
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer


def draw_sigmoid():
    x = np.arange(-10.0, 10.0, 0.02)
    y = 1 / (1 + np.exp(-x))
    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max())
    plt.plot(x, y, color="blue", lineWidth=2.5, linestyle="-")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Sigmoid(x)")
    plt.show(math.sigm)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
    # iris = load_iris()
    # logreg = linear_model.LogisticRegression(C=1e5)
    # logreg.fit(iris['data'], iris['target'])
    # sample = np.array([[6.4, 3.1, 5.5, 1.8], [5.1, 3.7, 1.5, 0.4], [5.9, 3.0, 4.2, 1.5]])
    # print(logreg.predict_proba(sample))
    # print(logreg.predict(sample))
    # raw = sigmoid(np.dot(sample, logreg.coef_.T) + np.tile(logreg.intercept_, (3, 1)))
    # raw /= raw.sum(axis=1)
    # print(raw)
    # for row in sample:
    #     print(sigmoid(np.dot(logreg.coef_,sample.T)+np.tile(logreg.intercept_, (3,1))))
    # print(np.dot(logreg.coef_,sample.T)+np.tile(logreg.intercept_, (3,1)))
    tags = [
        "python, tools",
        "linux, tools, ubuntu",
        "distributed systems, linux, networking, tools",
    ]
    # python tools linux ubuntu distributed systems networking
    vec = CountVectorizer()
    data = vec.fit_transform(tags).toarray()
    print(data)
    t = [["python", "tools"], ["linux", "tools", "ubuntu"]]
    tt = [' '.join(ttt) for ttt in t]
    print(tt)
    print(1)

