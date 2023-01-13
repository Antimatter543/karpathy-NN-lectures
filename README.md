Very rough notes of whatI've done so far:
Video 1 - micrograd -- A bare implementation of neural networks. Created a Value wrapper class, implemented binary operations and backwards calls to allow a backwards call through mathematical expressions. Built neurons, layers, and MLP classes and forward passed data inputs with targets. Called backwards on the value, updated weights; achieving gradient descent and thus a neural net.

Video 2 -- makemore -- a character level bigram model for name generation. Evaluated using negative log likelihood loss, then switched to constructing a neural network (in this case, effectively one layer of 27 neurons, each neuron having 27 weights). Converted characters to one hot encoded vectors, then converted logits to probability distributions by exponentiating and normalising (softmax). I and optimised weights w.r.t minimising nll loss function during gradient descent stage.  mplemented all ideas and steps by hand. Pytorch, jupyter, matplotlib.

