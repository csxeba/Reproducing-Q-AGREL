"""
NumPy implementation of Q-AGREL described in https://arxiv.org/pdf/1811.01768v1.pdf
Currently fails to converge. WIP.

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

from util import pull_mnist, shuffle, softmax


def relu(z):
    return np.maximum(0., z)


def relu_b(z):
    """Subderivative of ReLU"""
    return (z > 0.).astype(float)


def glorot_kernel(shape):
    """This works better in the Keras code"""
    limit = 6. / np.prod(shape)
    return np.random.uniform(-limit, limit, size=shape)


def run_q_agrel(batch_size, epochs, learning_rate, ε,
                num_in, num_h1, num_h2, num_out, momentum):

    np.random.seed(1337)
    possible_actions = np.arange(10)

    # Glorot works better for me
    # U = np.random.uniform(low=-0.05, high=0.05, size=(num_in, num_h1))
    # V = np.random.uniform(low=-0.05, high=0.05, size=(num_h1, num_h2))
    # W = np.random.uniform(low=-0.05, high=0.05, size=(num_h2, num_out))

    U = glorot_kernel([num_in, num_h1])
    V = glorot_kernel([num_h1, num_h2])
    W = glorot_kernel([num_h2, num_out])

    # Wb = np.random.uniform(low=-0.05, high=0.05, size=(NUM_OUT, NUM_H2))
    # Vb = np.random.uniform(low=-0.05, high=0.05, size=(NUM_H2, NUM_H1))
    # Ub = np.random.uniform(low=-0.05, high=0.05, size=(NUM_H1, NUM_IN))

    # Note that these are views, so updating them modifies the original arrays as well.
    Wb = W.T
    Vb = V.T
    # Ub = U.T  # These are not used, they would be used to backpropagate into network input.
    # We use reciprocal views, even though they brake biological plausibility IMO.

    # Momentum velocities
    vU = None
    vV = None
    vW = None

    lX, lY, tX, tY = pull_mnist()

    rewards_sum = []
    N = len(lX) // batch_size

    for e in range(1, epochs+1):
        lX, lY = shuffle(lX, lY)
        if e == 500:
            print(" Dropped the learning rate!")
            e /= 10.
        print("\nEpoch {}".format(e))
        for i, (x, y) in enumerate(([lX[s:s + batch_size], lY[s:s + batch_size]]
                                   for s in range(0, len(lX), batch_size)), start=1):
            m = len(x)  # get current batch's actual size (might be less than <batch_size>)
            α = learning_rate / m  # incorporate it into the learning rate

            ################
            # Forward pass #
            ################
            A1 = x @ U     #
            Y1 = relu(A1)  #
            A2 = Y1 @ V    #
            Y2 = relu(A2)  #
            Q = Y2 @ W     #
            ################

            ####################
            # Action selection #
            ####################
            roll = np.random.random(size=m) < (ε ** e)
            S = np.argmax(Q, axis=1)
            P = softmax(Q[roll])
            S[roll] = [np.random.choice(possible_actions, p=p) for p in P]

            Z = np.zeros_like(Q)  # Z is one-hot, indicating the selected action
            Z[range(m), S] = 1.
            ####################

            ######################
            # Reward calculation #
            ######################
            R = (y == S).astype(float)  # Reward is 1 at right predictions
            𝛿 = Q[range(m), S] - R[S]  # RPE (reward prediction error)
            # 𝛿 = R[S] - Q[range(m), S]  # TODO: unsure about the order
            E = (0.5 * 𝛿**2).mean()  # Integral of RPE, turns out to be MSE :)
            ######################

            #################
            # Backward pass #
            #################
            ΔW = α * (𝛿 * Y2.T) @ Z

            fbY2 = Z @ Wb  # g(O) term is here in the paper, but that equals to Z.
            g2 = (Y2 > 0).astype(float)  # This is actually the subderivative of ReLU!
            gbY2 = g2 * fbY2  # Merge these so this won't be calculated twice
            ΔV = α * (𝛿 * Y1.T) @ gbY2

            fbY1 = gbY2 @ Vb
            g1 = (Y1 > 0).astype(float)
            gbY1 = g1 * fbY1
            ΔU = α * (𝛿 * x.T) @ gbY1
            #################

            ####################
            # Optimize weights #
            ####################
            if vU is None:
                vU = ΔU
                vV = ΔV
                vW = ΔW
            else:
                vU *= momentum
                vU += ΔU
                vV *= momentum
                vV += ΔV
                vW *= momentum
                vW += ΔW

            U -= vU
            V -= vV
            W -= vW
            # Ub -= ΔU.T
            # Vb -= ΔV.T
            # Wb -= ΔW.T
            ####################

            reward = R.mean()
            rewards_sum.append(reward)
            progress = i / N
            print("\r\t{:>7.2%} - acc {:.4f} - 'loss' {:.4f}".format(
                progress, reward, E.mean()), end="")
        print()

    plt.plot(rewards_sum)
    plt.grid()
    plt.show()


def main():
    EPOCHS = 50
    BATCH_SIZE = 1000
    LEARNING_RATE = 1.0
    MOMENTUM = 0.9
    EPSILON = 0.5
    NUM_IN = 784
    NUM_H1 = 300
    NUM_H2 = 100
    NUM_OUT = 10

    run_q_agrel(BATCH_SIZE, EPOCHS, LEARNING_RATE, EPSILON, NUM_IN, NUM_H1, NUM_H2, NUM_OUT, MOMENTUM)


if __name__ == '__main__':
    main()
