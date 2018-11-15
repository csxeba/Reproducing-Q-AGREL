import numpy as np
np.random.seed(1337)

from matplotlib import pyplot as plt

from keras.models import Model
from keras.layers import Input, Dense
from keras.datasets import mnist
from keras.utils import to_categorical, normalize
from keras.optimizers import SGD

from csxdata.utilities.vectorop import shuffle


def softmax(z):
    ez = np.exp(z - np.max(z, axis=1, keepdims=True))
    return ez / np.sum(ez, axis=1, keepdims=True)


(lX, lY), (tX, tY) = mnist.load_data()
lX, tX = map(lambda x: x.reshape(-1, 784), [lX, tX])
lX, tX = map(normalize, [lX, tX])
lY, tY = map(to_categorical, [lY, tY])

# kernel_initializer = RandomUniform(-0.05, 0.05, 1337)
inputs = Input(shape=[784])
stream = Dense(300, activation="relu", use_bias=False)(inputs)
stream = Dense(100, activation="relu", use_bias=False)(stream)
Qs = Dense(10, activation="linear", use_bias=False)(stream)
model = Model(inputs, Qs)
model.compile(SGD(1.), "mse")

possible_labels = np.arange(10)
N = len(lX)
num_updates = N // 1000
losses = []
accs = []
for e in range(1, 31):
    print("\nEpoch", e)
    lX, lY = shuffle(lX, lY)
    for i, (x, y) in enumerate(([lX[s:s+1000], lY[s:s+1000]] for s in range(0, N, 1000)), start=1):
        q = model.predict(x)
        p = softmax(q)
        s = np.argmax(q, axis=1)
        roll = np.random.random(len(q)) < (0.5**e)
        s[roll] = [np.random.choice(possible_labels, p=prob) for prob in p[roll]]
        gt = np.argmax(y, axis=1)
        r = (s == gt).astype("float32")
        target = q.copy()
        target[range(len(target)), s] = r
        loss = model.train_on_batch(x, target)
        acc = r.mean()
        losses.append(loss)
        accs.append(acc)
        print("\r{:>7.2%} - E {:.4f} - A {:>6.2%}".format(i/num_updates, loss, acc), end="")

fig, (tax, bax) = plt.subplots(2, 1, sharex="all")
tax.plot(losses)
tax.set_title("Loss")
tax.grid()
bax.plot(accs)
bax.set_title("Accuracy")
bax.grid()
plt.show()
