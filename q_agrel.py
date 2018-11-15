import numpy as np
from matplotlib import pyplot as plt

from csxdata.utilities.loader import pull_mnist_data, shuffle


def softmax(z):
    ez = np.exp(z - z.max(axis=1, keepdims=True))
    return ez / ez.sum(axis=1, keepdims=True)


def relu(z):
    return np.maximum(0., z)


def relu_b(z):
    return (z > 0).astype(float)


def glorot_kernel(shape):
    limit = 6 / np.prod(shape)
    return np.random.uniform(-limit, limit, size=shape)


def run_q_agrel(batch_size, epochs, α, ε,
                num_in, num_h1, num_h2, num_out):

    possible_actions = np.arange(10)
    np.random.seed(1337)

    # U = np.random.uniform(low=-0.05, high=0.05, size=(num_in, num_h1))
    # V = np.random.uniform(low=-0.05, high=0.05, size=(num_h1, num_h2))
    # W = np.random.uniform(low=-0.05, high=0.05, size=(num_h2, num_out))

    U = glorot_kernel([num_in, num_h1])
    V = glorot_kernel([num_h1, num_h2])
    W = glorot_kernel([num_h2, num_out])

    # Wb = np.random.uniform(low=-0.05, high=0.05, size=(NUM_OUT, NUM_H2))
    # Vb = np.random.uniform(low=-0.05, high=0.05, size=(NUM_H2, NUM_H1))
    # Ub = np.random.uniform(low=-0.05, high=0.05, size=(NUM_H1, NUM_IN))

    # Note that these are views, so SGD on them modifies the original arrays as well.
    Wb = W.T
    Vb = V.T
    Ub = U.T

    lX, lY, tX, tY = pull_mnist_data()
    lX, tX = lX / 255., tX / 255.

    rewards_sum = []
    N = len(lX) // batch_size

    for e in range(1, epochs+1):
        lX, lY = shuffle(lX, lY)
        if e == 500:
            print(" Dropped the learning rate!")
            e /= 10.
        print("\nEpoch {}".format(e))
        for i, (x, y) in enumerate([lX[s:s + batch_size], lY[s:s + batch_size]]
                                   for s in range(0, len(lX), batch_size)):
            m = len(x)  # get current batch's actual size (might be less than <batch_size>)
            α /= m  # incorporate it into the learning rate

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
            roll = np.random.random(size=m) < ε
            S = np.argmax(Q, axis=1)  # m
            P = softmax(Q[roll])
            S[roll] = [np.random.choice(possible_actions, p=p) for p in P]

            Z = np.zeros_like(Q)  # Z is one-hot indicating the selected action
            Z[range(m), S] = 1.
            ####################

            ######################
            # Reward calculation #
            ######################
            gt = np.argmax(y, axis=1)  # Ground truth -> y is onehot
            R = (gt == S).astype(float)  # Reward is 1 at right predictions
            # 𝛿 = R[S] - Q[range(m), S]  # RPE (reward prediction error)
            𝛿 = Q[range(m), S] - R[S]  # RPE (reward prediction error)
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
            U -= ΔU
            V -= ΔV
            W -= ΔW
            # Ub -= ΔU.T
            # Vb -= ΔV.T
            # Wb -= ΔW.T
            ####################

            reward = R.mean()
            rewards_sum.append(reward)
            progress = (i+1) / N
            print("\r\t{:>7.2%} - R {:.4f} - E {:.4f}".format(
                progress, reward, E.mean()), end="")

    plt.plot(rewards_sum)
    plt.grid()
    plt.show()


def main():
    EPOCHS = 20
    BATCH_SIZE = 1000
    LEARNING_RATE = 0.5
    EPSILON = 0.5
    NUM_IN = 784
    NUM_H1 = 300
    NUM_H2 = 100
    NUM_OUT = 10

    run_q_agrel(BATCH_SIZE, EPOCHS, LEARNING_RATE, EPSILON, NUM_IN, NUM_H1, NUM_H2, NUM_OUT)


if __name__ == '__main__':
    main()
