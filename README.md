# Karpathy's Zero to Hero NN Series

Check out the notebooks and my code to see what I've actually written!


Rough notes of what I've done so far:




### Video 1 - Micrograd
A bare implementation of neural networks. Created a Value wrapper class, implemented binary operations and backwards calls to allow a backwards call through mathematical expressions. Built neurons, layers, and MLP classes and forward passed data inputs with targets. Called backwards on the value, updated weights; achieving gradient descent and thus a neural net.

### Video 2 -- Makemore
A character level bigram model for name generation. Evaluated using negative log likelihood loss, then switched to constructing a neural network (in this case, effectively one layer of 27 neurons, each neuron having 27 weights). Converted characters to one hot encoded vectors, then converted logits to probability distributions by exponentiating and normalising (softmax). I and optimised weights w.r.t minimising nll loss function during gradient descent stage.  mplemented all ideas and steps by hand. Pytorch, jupyter, matplotlib.

### Video 3 - Makemore NLP 
Like makemore, but we went into depth about creating multiple layers and generalising to a *context size* -- how many characters we use to predict the next one. Also embedded characters.
Effectively recreated the model from the [Bengio et al. 2003 MLP language model paper](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf).
<!-- ### Colab notebooks
| Colab Page | Video
| --- | --- |
### Video 3 - Makemore NLP 
Like makemore, but we went into depth about creating multiple layers and generalising to a *context size* -- how many characters we use to predict the next one. Also embedded characters.
Effectively recreated the model from the [Bengio et al. 2003 MLP language model paper](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf).
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camenduru/stable-diffusion-webui-colab/blob/main/stable_diffusion_webui_colab.ipynb) stable_diffusion_webui_colab | Micrograd -->