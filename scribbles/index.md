# Scribbles



## Transformers 

To deal with sequential data we have to options:
   - 1-D convolution NN
       - processing can be parallel 
       - not practical for long sequences 
   - Recurrent NN 
       - can't happen in prallel 
       - have gradient vanshing problem of the squence becomes so long
       - we have bottleneck at the end of the encoder 
   - RNN with attention mechanism 
       - to solve the bottleneck problem, we make Encoder-Decoder attention 
       - Decoder utilzes:
           - context vector 
           - weighted sum of hidden states (h1,h2, ... ) from the encoder 

### Transformers 

#### Encoder
- first we do input embedding, and positional embedding 
- in self attention: we multiply q,w,v by a matrix to do lenear transformation
- self attentoion: k * q --> scaling down --> softmax --> * v 
 
### multi-head  attention 
- works as we use many filters in CNN  
- in wide attention: it takes every word and spread it multi-head attention
- in narrow attention:  it take every word and split it up across the multi-head 
    - but didnt we lose the adcantage of using multi-head as mutli prespectives, as we do with filters in CNN?

### Positional info

- positional encoding using the rotation sin/cos matrix 
- positional embedding 

### Residual connections
- to give the chance to skip some learning parameters if it's better to minimize the loss 

### Layer Normalization 

- in batch normalization 
    - ==> we normalize to zero mean and unity varince 
    - we calculate for all samples in each batch (for each channel )
- in layer normalization 
    - ==>  $y = \gamma * x +  \beta $  where gamm and bata are trainable parametes
    - calculates  for all  channles in the same sample 
- in instance normalization ==> calculate for one channel in one sample 


## Debugging ML Models 

- Understand bias-variance diagnoses

    - getting more data ==> fixes high variance 
    - smaller set of features ==> fixes high variance 
    

#### Refrence
   - [Prof. Andrew NG Vid](https://www.youtube.com/watch?v=ORrStCArmP4)

## SVM and Kernels

* The main idea of kernels, is that if you can formlate the optimization problem as some of inner products of feater vectors, that can have infinite dimentions, and to come up with a way to calc these inner products efficiently 

we have $ X(i) \in R^{100}$, suppose  W can be expressed as a linear combintaion of X

$ W = \sum_{i = 1}^{M} \alpha_{i} y^i x^i$   (This can be proved with the representer theorem)
- vector W is perpendicular to the decsion boundry specified by algorithm, so W kinds of sets the orientation of the decision boundry and the bias moves it alont right and left.

optimization problem is :
$\min  {w,b}  {1/2} *||W||^2 $  
s.t  $y^i*((W^T * x^i) + b) >= 1$

* For SVM you can make a trade off between the margin and how much you can tolerate wrong calssified examples using a constant 


## Distributed Training  in Pytorch

### Pytorch DDP Internal 

DDP relies on c10d ProcessGroup for communications. Hence, applications must create ProcessGroup instances before constructing DDP.
The DDP constructor takes a reference to the local module, and broadcasts state_dict() from the process with rank 0 to all other processes in the group to make sure that all model replicas start from the exact same state.
DDP registers autograd hooks during construction, one hook per parameter. These hooks will be triggered during the backward pass when the gradient becomes ready.

Backward pass: Because `backward()` function is called on the loss directly, which out of DDP's control. So, it waits till one of the autograd hooks are invoked, to trigger the gradients synchronization.

DDP waits for all gradients in one bucket are ready, Reducer kicks off an asynchronous allreduce on that bucket to calculate mean of gradients across all processes.

Optimizer Step: From the optimizer’s perspective, it is optimizing a local model. Model replicas on all DDP processes can keep in sync because they all start from the same state and they have the same averaged gradients in every iteration.


### DataParallel VS DistributedDataParallel

- `DataParallel` is single-process, multi-thread, and only works on a single machine, while `DistributedDataParallel` is multi-process and works for both single- and multi- machine training. DataParallel is usually slower than DistributedDataParallel even on a single machine due to GIL contention across threads, per-iteration replicated model, and additional overhead introduced by scattering inputs and gathering outputs.
- `DataParallel` doesn't support model parallel 
    




#### Resources 
    - https://pytorch.org/docs/master/notes/ddp.html
    - https://pytorch.org/tutorials/intermediate/ddp_tutorial.html?highlight=distributed%20training

## Complete Statistical Theory of Learning - **Vladimir Vapnik**
* There are two ways for generalization: more data, and complete learning theory
* Turing said, that you should imitate intelligent person 

### Ref: https://www.youtube.com/watch?v=Ow25mjFjSmg

## Statistical Machine Learning
### part 1: 
#### deduction vs induction:
* deduction: is the process of reasoning from one or more general statements to reach a logically certain conclusion. premises must be correct.
* induction:  reasoning that construct or evaluates general proposition that are derived from specific examples. we can never be sure our conclusion can be wrong! 
* machine learning tries to automate the process of inductive inference.  
#### why should ML work?
* ML tries to find patterns in data 
* we will only be able to learn if there's something to learn
* ML makes some assumptions, which are are rarely made explicit.
* we need to have an idea what we are looking for. This is called "inductive bias". Learning is impossible without such a bias
    * the formal theorem if this is called `no free lunch theorem`
* on the other hand, if we have a very strong inductive bias, then with just few training examples, then we can have high certainty in the output
* the problem of selecting a good hypothesis class is called model selection.
* any system that learns has an inductive bias.
* if the algorithm works, **THERE HAS TO BE A BIAS**
* the inductive bias, rules over our function space 
### Part 2:
* it;s not hard for ML algorithm to correctly predict training labels
* usually ML algorithms make training errors, that is the function,they come up with doesn't perfectly fit the training data
* what we care about is the performance on teh test set
* it's not always the case that lowering the train data would lower that test data
#### K-Nearest algorithms:
* for K-nearest algorithm, 
    - the best value for K is log(N)
    - if k is too  small ==> overfitting 
    - if k is too large ==> underfitting
* k nearest algo achieves good results on MNIST dataset for classifying two classes, with simple euclidean distance function
* k-nearest can be used for density estimation, clustering, outlier detection 
* the inductive bias in K nearest algo, is that near points are the of teh same category 
* the challenging part about k nearest algo is how to measure the distance between points

### Part3
* for ML, we don't put any assmuptions for the data probability ditribution
* often, the input and output are random variables 
* in some applications, it's important that the loss depends on the input X.
* also in some cases, the type of error is critical, for exp. spaam detection 
* Bayes risk: is the min error for the expected values over all examples --> basically the it's the lowest error you can achieve
* Consistency: we say algorithm A is consistent, if we have an infinite iid datapoints, and the risk of algorithm's selected function converges to the Baye's risk. 
    - basically means that if we have infinite data samples, then our algorthms reaches the Bayes risk, which is the lowest error possible 
* Universally consistent: no mattter the underlying probability distribution is, when we have enough data points, the algorithm would be consistent
    - consistent independantly of the data distribution
    - KNN classifier, SVM, boositn, random forests are universally consistent

## DL Book 
### CH1
* one of the key ideas in DL is that data representation matters a lot, and that DL is a technique for learning how to represent the data

### AI 
* in AI we need the computer to do some tasks like humans
* we do that by providing the computer with a lot of rules describing the world and how to act in different scenarios
### ML
* in machine learning we can learn these rules without explicitly told them
* but we still need to be provided with custom features that are given by domain experts 

### Representation learning
* a specific type of ML where we don't tell the computer the specific features
* instead, we give the computer raw input, and it should learn the more complex features explicitly
* ex: autoencoders 
### DL 
* is a representation learning algorithms that is applied in multi sequential manner

![](learning-paradigms.png)

### CH2


## Deep Generative Modeling

* latent variable: it's a variable that is controlling some behaviors, but we can't directly observe it
* we are trying to observe `true explanatory factors`, for example, `latent variables`, from only observed data 

### Autoencoders:
* the encoder learns to map from data, to a low-dimensional latent space 
* the decoder learns to map back from the low-dimensional space back into a reconstructed observation 
* the bottleneck hidden layer forces the network to learn a compressed latent representation
* the reconstruction loss forces the latent representation to capture as much information from the data

### Variational Autoencoders (VAE)
* with classic autoencoders, once we train the network, then the latent representation is deterministic
* but in VAE, we add some randomness, so we can generate new samples 
* so the encoder should output a mean and a standard deviation, which represents a distribution of the input, then we can sample from this distribution to generate new sample 
* the encoder is trying to infer a probability distribution of the latent space with respect to its input data
* the decoder is trying to infer a new probability distribution over the input space given the latent space 
* the loss is going to be function of the parameters of the two distributions 
* the loss would consist of a construction loss and a regularization term, which is responsible for inducing some notion of structure of this probabilistic space 
* We need regularization and a prior to:
    - continuity: points that are close in the latent space, should remain close after decoding
    - completeness: samples from the latent space, should produce meaning content after decoding 
* we can't perform back propagation, as there's stochasticity in the latent space,
* to solve this issue, we fix, the mean and variance, and introduce the stochastic term separate from them 

* The key problem with VAEs, is a concern of density estimation 

### Generative Adversarial Networks (GANs)
* we need to sample from a very complex distribution, that we don't know, and can't estimate
* the solution, is to sample from a simple distribution (eg. noise), then learn a transformation, to the data distribution  
* we have a Generator, that's tries to transform the data sampled from the random noise, into data that looks real, to trick the discriminator
* we have a discriminator, which tries to identify real data from fake. 

## SubWords 
* it's just like the skip-gram model
* but we just changed the score function
* so that we increased the vocab size by adding N-grams of all words we have
* then we use them to capture more meaning from the words
* if we encounter new word, then we add it's N-gram and thats would be the word vector 
## Decision Trees
* they are greedy algorithm
* they can stuck in local minimum 
* if the we have some continuous features, we can use it multiple times, every time with different range 
## Purity function
* we want to define a purity function, that has the following
    - it has it's maximum value when probability of any class in 1 
    - it has it's minimum value when all classes has the same probability
    - Purity( pa, pb)  == Purity (pb,  pa)
*  entropy = impurity  = -1 * purity 
* one function that satisfies all these requirements, is 

    - $ purity(p_{c1}, p_{c2}, p_{c3}) =  p_{c1} \log(p_{c1}) * p_{c2} \log(p_{c2}) * p_{c3} \log(p_{c3})$
* so we choose features, that would increase purity the most after splitting the dataset using it 
* to calculate after entropy or purity of a set after seperation, would be the weighted average of the subsets 

## Why going deep in Deep Learning
* one motivation, is that going deep can reduce the size of our units exponentially
* in our underlying function, there could be some symmetry, that we can fold the function across its axis
* for statistical reasons, we would want to infer out initial beliefs, about our function, that is involve the composition of several simpler functions
* Empirical experiment[](https://www.youtube.com/watch?v%3DKbBMaMVk0eE)s show that deeper networks generalize better 


## Few-shot Learning

* we want to classify examples, that we only have few examples for, maybe even 0
* the idea is instead of having a classifier as the last layer(softmax layer), we can use a siamese network, just to tell us is the two examples are similar
* so we just learn a similarity function 

## GLIDE: 
* Generates realistic images from text
* GLIDE is the next version of DALLE, with respect to photo realism and caption similarity 
* this is fine-tuned model, not zero-shot like DALLE 
* It can generate or edit images 
* so you can generate an image using zero-shot, then you can edit the image by putting masks on the image and tell the model what to draw in the masked area 

### Diffusion Models

* we start with the original image, then we keep add noise to it till it become so noisy
* then we try to reverse the operation and get it back to the input image  


## P vs NP

* P = problems solvable in polynomial time
* NP = decision problems solvable in nondeterministic polynomial time
    - decision problems = yes, no problems 
    - NP problems are hard to solve, but each to check the correctness of the answers 

