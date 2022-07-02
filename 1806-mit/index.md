# 

# MIT 18.06 Linear Algebra course

---
> MIT 18.06 Linear Algebra course

- toc: true 
- badges: true
- comments: true
- categories: [jupyter,MIT,Linear Algebra, Math]
--- 
## Lecture 1

We learn about the big picture behind mutplication of matrix and vector 

we learn about the row picture and column picture 


## Lecture 2

we learned about elimination method to solve a system of equations 


## Lecture 3

in this lecture  we learned about matrics multiplication:

we can do that in five ways: 

1. row * col ==> gives an entry (1 cell)
2.  col * row ==>   sum (  r1 * c1 ,   r2 * c2, etc)
3. by columns  ==>  A * c1  = combination of A columns 
4. by columns  ==>  r1 * B  = combination of A B rows 
5.  by blocks  ==>   A (A1,A2,A3,A4)  *  B (B1,B2,B3,B4) =  C1 = (A1* B1 +  A2 * B3)  and so on



then we learned about  gausian-Jordan  elimination to find the matrix inverse 

[A | I] ==> we apply elimination till we  get [ I | A-1  ]




## Lecture 4

in this lecture we learn about  A= L U, where L is E^ -1, and whats special about this is that it has all multipliers in the lower triangular  with ones on the diagonal   

## Lecture 5

we continued a little with permutations and moved to vector spaces 
 we learned about  sub spaces and columns spaces  ==> which is u take  the columns of the matrix and  all its combinations  and that creates a plane through origin making a columns space 



## Lecture 6

In this lecture we continued about columns spaces and that we build those up by taking the combinations of all columns.
Then we learned about null spaces while are  sub spaces of X that satisfies   A X = 0

## Lecture 7

in this lecture we  continued about null space 

 then we learned about  the special solution, where we assume the variables of the free vector then  get the special solution 

finally we learned about the reduced form 

where R =  [             I   F      
                                0   0      ]

and the null matrix is     [     -F 
                                               I     ]

then R N = 0

## Lecture 8

in this lecture we expanded to talk about  A x = b 

 and we find the whether there's a solution to the equation or not  depends on the rank of the matrix  

 also we get the  Xcomplete  = Xparticular + Xnull space 
and we get particular soln by putting all free variables = zero 


![image.png](attachment:image.png)


## Lecture 9

in this lecture we learned about independent columns and how they make a space, we also learned about Basis and what are two conditions for it 


rank(A) = number of pivot columns of A = dimension of C(A). 


dimension of N(A) = number of free variables = n − r, 


## Lecture 10

In this lecture we learned about the four subspaces 

we also  started in matrix space M

## Lecture 11

We learned about matrix space 
we  take introduction about graph


## Lecture 12

in this lecture we learned about graphs and how to represent them with matrices, then we applied that to electrical system and applied kerchofs law 

## Lecture 13

Quiz 1 review


## Lecture 14

in this lecture learned about othrignilaity of the four vector spaces and what does it means 



## Lecture 15

in this lecture we learned about projection  of matrices into subspaces 


## Lecture 16


we got example explaning the  projection into subspaces and how to get the best fit using the least square error 


## Lecture 17

in this lecture le learned about orthonormal vectors and their special features and  we learn how to produce them from any independent vectors  using gram-schmeit  


## Lecture 18

Propertise of determants



## Lecture 19

- det I =1 
- sign reverses with each row or colums exchange 
- det is linear in each row seperately

### Big Det Formula




for a N * N matrix, we calc the sum of N! terms
 
$$detA=\sum_{i=1}^ N a1\alpha*a2\beta*a3\gamma*an\omega$$


(where $\alpha, \beta, ... \omega $) = perm of (1,2,3, ..., N )

### Cofactors 

    cofactor of aij  =  Cij  =   +/-  det of ( n-1 matrix with column j, and  row i erased )
        it is plus if i+j is even, minus if i+j is odd 

#### cofactor formula (along row 1)
    det A = a11 C11 + a12 C12 + .... + a1n C1n

## Lecture 20 


$ A^{-1} = 1/detA * C^T$  where C is the cofactors matrix

### Cramers rule 

A x = b

x= A^ (-1) b  = 1/detA C^T b 

X_j = detB_j / detA  where B_j is A matrix with column j replaced by b


### Det A = Volume 

detA = volume of the shape created by making an edge from each of the rows




## Lecture 21 Eigenvalues and Eigenvectors

Eigenvectors:
 Ax is prallel to x  ==> Ax =  $\lambda$x
 
 lambda ia the eigen values
 
 if we have a plane: 
 - any x in the plane: Px= x ==> x is eigenvector and lambda = 1
 - any x perpendicular to plane Px = 0 ==> x is eigen vector and lambda = 0 
 
 
#### Fact: the sum of the eigenvalues = the sum of the diagonal of A 

## Lecture 22: Diagnolization 


to get power of matrix $A^k$
- first get the eigenvalues and vectors for A


- then compute $A = S \lambda S^{-1}$ where S is the eigenvector matrix, and Lambda is diagonal matrix of the eigenvalues 

- then $A^k = S * \lambda^k*S^{-1}$


## Lecture 23  



for the diffrential equations:

1- Stability
if  lambda < 0 ==> u(t) --> 0 

2- Steady state
 if lambda1 = 0, lambda2 < 0

3- Blowup  if  any lambda > 0 

## Lecture 24


### Markov Matrix

1- All entries >= 0
2- The sum of every column is 1 

3- lambda = 1 is eigenvalue 

4- all othe lambda < 1 
5- eigenvector values  >= 0 


### Fourier series 

integration of  (  f(x) g(x) dx   ) from 0 to 2pi =  0 

## Lecture 25 

### Symmetric matrices 

$ A  = A^T$

- the eigenvalues are real 
- the eigenvectors are perpendicular 


usual case: 

 $A  = S \lambda S^{-1}$
 
 symmetric case:
  - we have orthonormal eigenvectors 
  
$A  = Q \lambda Q^{-1}  = Q \lambda Q^{-T} $

### **Every symmetric matrix is a combination of perp. projection matrices** 

### **Signs of pivots are the same as the sign of the eigenvalues** 

**product of pivots   = product of eigenvalues =  det of matrix**

## Positive definite symmetric matrix 

- all eigenvalues are positives 
- all pivots are positive
- det is positive as it's the product of the eigenvalues
- also all sub detemants are positive ( determants of lower matrices )

- if S is pos. definite ==> $ X^T*S*X$ must be positive 

## Lecture 26

### Complex Matrices 

 - we wanna utilize tha fact that $\bar{Z^T}*Z = \left\|{Z}\right\|^2$
 - Hermitian is biscally the conj. and transpose ==> $Z^H = \bar{Z^T}$

- Hermitian Matricies :  $A^H = A$

- perpendicular: q1, q2, ..., qn 
    - $\bar{qi}^T * qj = 0 if i!=j, 1  if i=j $
    - $Q^H*Q = I$
 


### Fourier Matrix

- a matrix with entries are powers of some number W. where  $W^n = 1$

- $  F^H*F = I$




### Fast fourier transform
- reduces complexity from $N^2  to N log(N)$
- $W_{2n}^2 = W_n  ==> W_4^2 = W_2$


## Lecture 27

when det= 0 ==> then the matrix is positive semi-definite


f(x1,x2,x3...,xn) ==> min when the matrix of second derivatives is positive definite

- the eigenvalues tells us the length of the axis of the shape crated by cutting through the shape of the $X^T A X $
- the direction of the eigenvectors is the direction of the axis of that shape 



## Lecture 28


A is a m by n matrix ==> $ A^T*A$ is a positive definite symmetric matrix

### similar matrices 

- A and B are similar means: for some M ==> $B = M^{-1}*A*M$
- **Similar matrices has the same eigenvalues if the eigenvalues are unique**
- the eigenvector of B  is  $M^{-1} * (eigenvector ofA)$

### Jordan form

- every square A is similar to Jordan matrix J 
- every jordan block has one eigenvaector
- the number of jordan blocks  = number of eigenvectors 

## Lecture 29

- eigenvalues of (AB) = eigenvalues of (BA)

### Singular value composition (SVD)

- $Av = \sigma u $
- $ A = u \sigma v^T  = u \sigma v^{-1}$ 
- $A^T A =  v \sigma^T u^Tu /sigma v^T = v \sigma^2 v^T$
- $A A^T=  u \sigma^T v^Tv /sigma u^T = u \sigma^2 u^T$


## Lecture 30

### Linear transformation 

- examples: projection, rotation

- if u know what transfotmation does to the basis of a plane, then u know what it does to every vector in the plane 
    - every $ v = c_1 v_1 + c_2 v_2 + ... + c_n v_n$
    - then $T(v) =  c_1  T(v_1) + .... + c_n T(v_n)$
    
- coordinates come from a basis (think of basis like the X-Y-Z axis  
- so we have a basis for the input and a basis for the output
- Rule to find matrix A, given the input and output basis :
    - input basis: v1 ===> vn
    - output basis: w1 ===> wm
    - 1st column of A : write T(v1)  = a11 w1 + a21 w2 + ... + am1 wm
    - 2nd column of A : write T(v2)  = a12 w1 + a22 w2 + ... + am2 wm
    - repeat that for the n columns 
- A * (input coordinates) = (output coordinates) 

## Lecture 31

### change of basis 
- we have a new basis vectors and we wanna change to the new basis W
- A = c * W ==> c = W^-1 * A
- when we change the basis, every vector would have new coordinates ==> old coordinates = new basis * new coordinates  ==> x = W c 


## Lecture 32 

### 2-sided inverse 
- $A A^{-1} = I = A^{-1} A$
- r = m = n  ==> full rank

### left inverse 

- full column rank
- r = n 
- nullspace = 0 
- then $A^T A $ is invertable 
- $A^{-1}_{left} =  (A^T A)^{-1} A^T $ 
- $A^{-1}_{left} A  = I$


### right inverse 

- full row rank
- r = m
- n-m free variables 
- left nullspace = 0 

### pseudo inverse $A^+$

$A^+ = v*\sigma^{-1}*u^T$

    


### Lecute 34 

- no solution ==> rank < m 
- has one solution ==> there'e no null space  ==> rank = n 
- a matrix is invertable when there's no null space ==> r = n  ==> indep. columns 
- positive definite matrix must have full rank ==> has no null space 
- positive def is invertable 
- the matrix has soln of any c when the matrix has full row rank 
- matix with orthogonal eigen vectors : 
    - symmetric matrices 
    - skew-symmetric 
    - orthogonal matrices 
- in markov matrix 
    - the eigen values are one, and some sother values less than one
    - $k_m$ and m goes to infinity  is the steady state we ge the eigenvector that corresponts to eigenvalue one asn multiply it with c, and notes that the sum of  u is alwayes the same, so the sum of u0 is the sum of uk, so look what c achieves that 


```python

```

