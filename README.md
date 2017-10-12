# WhatsCooking
Solution of What's cooking on Kaggle with an accuracy of **81.586%**

## classifier 1:

 * Treat ingredients as string.
 * Use TfidfVectorizer to vectorize training set and test set.
 * Use ensemble 25 neural networks to predict the result.
 
 ## classifier2:
 
  * Treat ingredients as string.
  * Use CountVectorizer to vectorize training set and test set.
  * Filter rare words whose occurance less than 5 times.
  * Generate new features by combining two words. Each type of cuisine we generate at most 200 new features.
  * Use TfidfTransformer on previous matrix.
  * Use neural network to ensemble several base models: LogisticRegression, StochasticGradientDescant, SupportVectorMachine, RandomForest, LightGBM, NeuralNetwork
  
  ## final classifier:
  
  The output of classifier 1&2 is a **9,944**(# samples in test set) by **20**(# of labels) matrix, whose entry (i, j) is the probability that the **i-th** sample in the test set blongs to **j-th** class.
  
  Then I calculate the mean value of two matrices, the predicted result of a sample **i** is **argmax<sub>j</sub>(prob<sub>ij</sub>**).
