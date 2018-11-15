# Reproducing Q-AGREL

Paper: https://arxiv.org/pdf/1811.01768v1.pdf

Q-AGREL is a biologically plausible alternative training method for Deep Learning models.

The standard method for the determination of neuron-level error (ie. gradient computation) is the
backpropagation of error algorithm, which (paired with Stochastic Gradient Descent) can produce
reasonably fast convergence times for deep neural networks.

One of the main criticism agains backprop is that it is not very likely to be the method for
learning in the brain, because it relies on the gradients of an external utility function.

Q-AGREL uses local information for the determination of a quasi-gradient, which is then
scaled by a global reward prediction error (a single scalar).

This repository contains an implementation of Q-AGREL in NumPy, and some experiments with the
algorithm in pure NumPy and in the Keras deep learning library.

The idea is not mine, nor am I an author of the paper, I am just someone fascinated by deep nets.
