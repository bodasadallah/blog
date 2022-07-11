# NLP Specialization 
> Notebook for coursera's NLP Specialization 

- toc: true 
- badges: true
- comments: true
- categories: [jupyter,deeplearning,python,NLP]


# Course 1: Classification and vector Spaces

## Weak 4 

### Hashing 

* We can use hashing to search for the K-nearest vectors, to heavily reduce the searching space

#### Locality senstive hashing 
* the idea is to put items that are close in the vector space, in the same hashing buckets 
* we can create a set of planes and calculate the relative position of points compated to this plane and then we can calculate the hash value for this point accordingly 

* but how can we be sure, that the planes that we chose are the perfect set to seperate our space, we can't be sure of that, so we create a multi sets of random planes and every set would get us a different way to seperate our words

# Course 2: Probabilstic Models 


## Week 1: Autocorrect

* to make a simple auto-correction system, you need to perform 4 simple steps:
    - you need to identify the miss spelled words
    - get the n edit distance correct words
    - filter these candidates for correct words in the dictionary
    - change miss spelled word with one that has the highest probability
    


## Week 2: POS using Viterbi algorithm

* Part of Speech Tagging (POS) is the process of assigning a part of speech to a word
* You can use part of speech tagging for: 
    - Identifying named entities
    - Speech recognition
    - Co-reference Resolution
* You can use the probabilities of POS tags happening near one another to come up with the most reasonable output.

### Markov Chains:
* You can use Markov chains to identify the probability of the next word.
* calculate transmission probability
* calculate emission probability

### Viterbi algorithm
* calculates a probability for each possible path
* a probability for a given state, is the (transition probability * emission probability)
* total probability of the path, is to product of all states probabilities 
* we use a top down dynamic programming algorithm to build the paths matrix 
* we use sum of logs instead of product of probabilities, to avoid converging to zero values  

## Week3 Autocomplete

### N-Gram models:
* it's a language model that predicts probabilities of sentences depending on the probabilities of their N-grams 
* to capture the context of beginning end ending of the sentences, we add start and end tokens to each sentence
### Preplexity:
* a measure to calculate how complex a sentence is
* humans type low preplex sentences

### out of vocab words and smoothing
* we can add UNK token for unseen words in the vocab
* we can apply smoothing of interpolation for unseen Ngrams 

## Week4: Word vectors using Bag of Words method

* we can use self-supervised learning in predicting the next word, to learn the weight matrix

* word2vec
    - continuous bag-of-words 
        - predicts a word in context
    - continuous skip-gram
        - tries to predict the words surrounding input word
* Global Vectors (GloVe)
    - factorizes the corpus word co-occurrence matrix
* fastText
    - based on skip-gram model
    - support out-of-vocab words 
* Advanced word embedding methods
    - Deep Learning, contextual embedding 
    - BERT
    - ELMO 
    - GPT-2
### Continuous Bag-of-Words Model
* you choose a center word, and a context words around it, and try to predict the centered word
* we model the words by one-hot encoding 
* then we prepare the input training example feature to be an average of the one-hot vectors of the context words, and the label would be one-hot vector of the center word
* after training the model, the word embedding is one of the weight matrices, or an average of them
#### Evaluations
-   Intrinsic evaluation
    - test the relationships between words
    - Analogies
    - Clustering
    - Visualizing
- Extrinsic evaluation 
    - test the embedding on the end task you want to perform (ex. Sentiment Analysis)
    - evaluates actual usefulness of embeddings
    - time consuming
    - more difficult to troubleshoot




