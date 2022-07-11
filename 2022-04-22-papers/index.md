# 

# Deep Learning Papers Summarization 
> A Summary of DL papers

- toc: true 
- badges: true
- comments: true
- categories: [jupyter,deeplearning,python]

## Decoupled Neural Interfaces using Synthetic Gradients

* In NN, the training process, has 3 bottle-necks
    - forward lock: you need to calculate teh output of the previous layer before you can can go into next layer in forward pass
    - backward pass: the same, but for backward propagation 
    - weights lock: you can't update weights unless you do for weights in next layer
* the paper trying to unlock these bootle-necks by decoupling each layer, to be sufficient alone
* it does that by introducing, a Synthetic Gradient Model, that can predict the gradient for the current layer, without waiting for the gradient of the next layer
* this was we can calculate gradient and update weights as soon as we calculate the activation of the current layer

### Synthetic Gradient Model
* can be just a simple NN that is trained to output the gradient of the layer
* it can be trained using the true gradient, or even the synthetic gradient of the next layer
* it's important that the last layer computes the true gradient, as in the end we must have a ground truth to can calculate a true loss, and the NN would actually train


* we can have also synthetic model for forward pass, that works with the same idea

## A Roadmap for Big Models

* We are in the Era of  Big Models
* Model generalization is hard, models trained on certain data domain, doesn't scare to other
* Datasets creation, and high research tasks, made it hard for small companies to train task-specific models
* Big models solve thees issues.

### Big Models
* Big-data driven
* Multi-task Adaptive 
* can fine-tuned with few-shot learning
#### Data issues
* data bias
* data duplication 
* data has to cover all domains
* low quality data
* hard to create huge datasets

### Knowledge 
* a new way to represent data
* we represent knowledge as knowledge graphs
* KG consists of: Instances, Relation, Concept, and Values
* KG can be created using : experts, wiki-based knowledge graphs, or extracted from unstructured texts
#### KG Completion and Integration
* most of the known KGs has many fields empty, and there's a going research in how to deal with that and fill the gaps.
* some methods try to do that using intra-graph knowledge augmentation or with inter-graph.



