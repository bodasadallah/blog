# Computational Linear Algebra


# Lecture 1

```python
import numpy as np
```


```python
a = np.array( [[6,5,3,1], [3,6,2,2], [3,4,3,1] ])
b = np.array( [ [1.5 ,1], [2,2.5], [5 ,4.5] ,[16 ,17] ])
```


```python
for c in (a @ b):
    print(c)
```

    [50. 49.]
    [58.5 61. ]
    [43.5 43.5]


# Lecture 2 

Matrix decomposition: we decopose matricies into smaller ones that has special properties 


### Singular Value Decomposition (SVD):
* it's an exact decomposition, so you can retrieve the orginal matrix again 

#### Some SVD applications: 
* semantic analysis
* collaborative filtering / recommendation 
* data compression 
* PCA (principal component analysis)


### Non-negative Matrix Factorization (NMF)




