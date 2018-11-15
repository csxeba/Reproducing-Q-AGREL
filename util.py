import numpy as np
from keras.datasets import mnist
from keras.utils import normalize


def pull_mnist():
    """Note that I'm not transforming Y to one-hot!"""
    (lX, lY), (tX, tY) = mnist.load_data()
    lX, tX = map(lambda x: x.reshape(-1, 784), [lX, tX])
    lX, tX = map(normalize, [lX, tX])
    return lX, lY, tX, tY


def shuffle(x, y):
    """Shuffles two ndarrays together"""
    arg = np.arange(len(x))
    np.random.shuffle(arg)
    return x[arg], y[arg]


def softmax(z):
    """Numerically stable-ish softmax :)"""
    ez = np.exp(z - np.max(z, axis=1, keepdims=True))
    return ez / np.sum(ez, axis=1, keepdims=True)


