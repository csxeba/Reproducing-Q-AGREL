"""
Keras-based reimplementation of some experiments of Q-AGREL described in https://arxiv.org/pdf/1811.01768v1.pdf
This basic level Keras is only able to run Q-AGREL with strictly reciprocal weights.

Script author: csxeba
author e-mail: csxeba {at} gmail {dot} com

Copyright (c) 2018, Csaba Gór

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""

import numpy as np

from matplotlib import pyplot as plt

from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import SGD
from keras.utils import to_categorical

from util import pull_mnist, shuffle, softmax

np.random.seed(1337)

lX, lY, tX, tY = pull_mnist()
tY = to_categorical(tY, num_classes=10)

# Below initialization scheme fails to converge for me
# kernel_initializer = RandomUniform(-0.05, 0.05, 1337)
inputs = Input(shape=[784])

# Bias doesn't really change the performance, so I switched it off to be more similar to the numpy script
stream = Dense(300, activation="relu", use_bias=False)(inputs)
stream = Dense(100, activation="relu", use_bias=False)(stream)
Qs = Dense(10, activation="linear", use_bias=False)(stream)
model = Model(inputs, Qs)
model.compile(SGD(1.), "mse", metrics=["acc"])

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
        # Action selection: we either take the highest Q greedily or sample from the softmax of Q.
        s = np.argmax(q, axis=1)
        roll = np.random.random(len(q)) < (0.5**e)
        s[roll] = [np.random.choice(possible_labels, p=prob) for prob in p[roll]]
        # Reward is 1 if prediction was right, 0 otherwise
        r = (s == y).astype("float32")  # Note that y is NOT onehot!
        # Minor hack, so we can use the convenient .train_on_batch(x, y) interface
        target = q.copy()
        target[range(len(target)), s] = r
        # Gradient is only computed for the output neuron, which did the prediction
        loss, _ = model.train_on_batch(x, target)
        acc = r.mean()
        losses.append(loss)
        accs.append(acc)
        print("\r{:>7.2%} - E {:.4f} - A {:>6.2%}".format(i/num_updates, loss, acc), end="")
    print()
    print("Validation acc: {:.2%}".format(model.evaluate(tX, tY, batch_size=32, verbose=0)[1]))

fig, (tax, bax) = plt.subplots(2, 1, sharex="all")
tax.plot(losses)
tax.set_title("Loss")
tax.grid()
bax.plot(accs)
bax.set_title("Accuracy")
bax.grid()
plt.show()
