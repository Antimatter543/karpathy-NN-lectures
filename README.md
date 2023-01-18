# Karpathy's Zero to Hero NN Series

Check out the notebooks and my code to see what I've actually written! Used numpy, pytorch (my tensors are flowing), jupyter, matplotlib.


## Snippets from my notebooks
That I found fun!
(like, the counts matrix, ..., the pltshow of my actual weights gradient -- shows what effect the weight has on the final loss! So crazy, and so cool.)
>> Do an open in collab for our 2 - makemore vid exercises?? I think that's a good representation of stuff we can do/we have been doing.

| Name | Jupyter File |  Colab Page
| ---  | --- | --- |
Makemore - Video 2 | [Click here](https://github.com/Antimatter543/karpathy-NN-lectures/blob/main/2%20-%20makemore/makemore.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Antimatter543/karpathy-NN-lectures/blob/main/2%20-%20makemore/makemore.ipynb)
Exercises - Video 2 | [Click here](https://github.com/Antimatter543/karpathy-NN-lectures/blob/main/2%20-%20makemore/vid_exercises.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Antimatter543/karpathy-NN-lectures/blob/main/2%20-%20makemore/vid_exercises.ipynb)


## Video summaries (by me)
<!-- Partially so I have an easier time putting these on my resume 😳 -->

### Video 1 - Micrograd
A bare implementation of neural networks. Created a Value wrapper class, implemented binary operations and backwards calls to allow a backwards call through mathematical expressions. Built neurons, layers, and MLP classes and forward passed data inputs with targets. Called backwards on the value, updated weights; achieving gradient descent and thus a neural net.

### Video 2 -- Makemore
A character level bigram model for name generation. Evaluated using negative log likelihood loss, then switched to constructing a neural network (in this case, effectively one layer of 27 neurons, each neuron having 27 weights). Converted characters to one hot encoded vectors, then converted logits to probability distributions by exponentiating and normalising (softmax). Optimised weights by minimising nll loss function during gradient descent stage. Created train, test, and dev sets for evaluation of models.    Implemented all ideas and steps by hand. Pytorch, jupyter, matplotlib.

### Video 3 - Makemore NLP 
Like makemore, but we went into depth about creating multiple layers and generalising to a *context size* -- how many characters we use to predict the next one. Also embedded characters, and focused on finetuning hyperparameters like learning rate, embedding size, number of layers, ... Effectively recreated the model from the [Bengio et al. 2003 MLP language model paper](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf). Worked on many basics of ML: model training, learning rate tuning, hyperparameters, evaluation, train/dev/test splits, under/overfitting, etc.).


### Video 4 - Makemore, BatchNorm
- Initialised the NN to have a much better initial loss (no hockey stick 'easy gains'!) (you should always have some rough idea of what initial loss should look like).

<!-- ### Colab notebooks
| Colab Page | Video
| --- | --- |
### Video 3 - Makemore NLP 
Like makemore, but we went into depth about creating multiple layers and generalising to a *context size* -- how many characters we use to predict the next one. Also embedded characters.
Effectively recreated the model from the [Bengio et al. 2003 MLP language model paper](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf).
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camenduru/stable-diffusion-webui-colab/blob/main/stable_diffusion_webui_colab.ipynb) stable_diffusion_webui_colab | Micrograd -->