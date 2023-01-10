# CS480/680 Intro to Machine Learning


## Lecture 12

### Gausain process

- infinite dimentional gaussian distribution

## Lecture 16

### Convolution NN

- a rule of thumb: to have many layers with smaller filters is better than having one big filter, as going deep captures better features and also uses fewer parameters

### Residual Networks

- even after using Relu, NN can still suffer from gradient vanishing
- the idea in to add skip connections so that we can create shorter paths

## Lecture 18

### LSTM vs GRU vs Attention

- LSTM: 3 gates, one for the cell state, one for the input, one for the output
- GRU: only two states, one for output, and one for taking weighted probablitiy for the contribution of the input and the hidden state
  - takes less parameters
- Attention: at every step of producing the output, create a new context vector that gives more attention to the importat input tokens for this output token

## Lecture 20

### Autoencoder

takes different input and generates the same output

used in:

- compression
- denoising
- sparse representation
- data generation

## Lecture 21

### Generative models

#### Variational autoencoders

- idea: train the encoder to sample a fixed distribution ,
- we want the network to sample a fixed distribution that is close to the distribution of the encoder, so that it generate similar outputs to the input, but not the same

#### GANS:

-

## Lecture 22

### Ensemble Learning:

- the idea is to combine the hypothesis of several models to produce a better one

#### Bagging

       - choose the class the majority votes

#### Weighted majority

    - decrease the weight of corrlelated hypothesis
    - increase the weight of good hypothesis

#### Boosting

    - idea: when an instance is missclassified  by hypothesis, increase its weight so that the next hypothesis is more likely to classify  it correctly

    - can boost a weak learner

    - makes weighted training set, so that it can focus on missclassified examples


    - at the end generate weighted hypotheses based on the acc of each hypothsis

    - Advantages:
        - no need to learn perfect hypothesis
        - can boost any weak learning algo
        - boosting is very simple
        - has good generalization


**Netflex challenge 2006**

### Lecture 23

#### Normalizing flows

### Lecture 24

#### Gradient boosting

- boosting for regression

