# COMP7404 Computer Intelligence and Machine Learning

### Group Project Proposal

## Group Members:
* 
* Mak Tak Hei 3035420273
* Lui Sin Ying Bianca 2010085577
* Lo Rocky 3035420077
* Chan Wing Hei 3035186243


## Content List

1. Gradient Boost, Gradient Boosted Tree, and its Implementation (preferred)
2. Decision tree ensembling and its implementation
3. Curse of Dimensionality, Remedies to Complexity


## 1. Gradient Boosting, Gradient Boosted Tree, and Its Implementation
#### 1.1 Description

With gradient descent, the machine learning algorithm attempts to find the parameter values which produce the minimal cost, but the architecture of the hypothesis is fixed.  On contrary, gradient boosting optimizes both the architecture and the parameters.  In this project, the approach of gradient boosting and its specialization of gradient boosted tree will be discussed.  One such implementation, XGBoost, will be demostrated to show its performance.

#### 1.2 Methodology
XGBoost is designed as a highly scalable end-to-end tree boosting system. It is a tree ensemble system which tries to give predictions by minimizing some regularized objective function. However, such objective function in tree ensemble system is not able to be optimized by traditional optimization techniques in the Euclidean space. Therefore, it is proposed to use the first and second order gradient statistics to optimize the functions, to use a tailor-made greedy algorithm to speed up the evaluation process by avoiding to enumerate all possible trees in the tree system, and to adopt the “Shrinkage and Column Subsampling” technique to avoid overfitting to the regularized objective.

#### 1.3 Demo to be Implemented
* Dataset:
   * Kaggle Competition: PLAsTiCC Astronomical Classification
   * https://www.kaggle.com/c/PLAsTiCC-2018/
* Reason of Choosing This Dataset
    1) Well-defined dataset, smaller effects to classification result from data integrity or data engineering
    2) Binary classification problem, a less complicated problem which avoids diminishments to classifier's performance due to complexity of the problem itself
    3) Works from other fellows, comparisons can be made on performances from different works
* Evaluation on Tree Strucutre Constructed
    1) Are the rules/tree structures constructed being useful/too general?
    2) Speed of constructing tree structures 
    3) Hyper-parameters to build the tree with optimal performance
    4) Comparisons to other submissions:
    5) Algorithm used
    6) Classification Accurary

## 2. Ensemble Learning - How to blend your models
#### 2.1 Description

Ensemble methods complement ML algorithms by combining several base learners to improve prediction accuracy. Sequential methods (reduce bias e.g. boosting) and parallel methods (reduce variance e.g. Random Forest) will be discussed in terms of intuition, algorithm and pros/cons. A new framework, Boosted Random Forest(BRM), combining the strengths of both methods will be demonstrated.

#### 2.2 Methodology
Boosting is an ensemble algorithm that combines weak learners to construct a classifier of higher discriminating performance by sequential training of classifiers in which the previous classifier’s classification errors is used to for subsequent training. However, boosting tends to overfit.

Random forest, however, construct a number of random and seemingly independent decision trees by bagging and feature selection which can avoid overfitting. The cost is that a large number of random decision trees must be constructed and many of them may not improve accuracy or even look similar. This makes the training slow and ineffective for large scale data.

To get the best of both world, “Boosted Random Forest” is suggested to reduce the number of decision tree but also get higher generalization ability compared to tradition random forest methodology. A demonstration will be given to compare its performance with the traditional random forest and boosted tree(base models) in terms of training time model size and testing error.

#### 2.3 Demo to be Implemented
* Dataset:
   * UCI Machine Learning Repository: Bank Marketing Data Set
   * https://archive.ics.uci.edu/ml/datasets/Bank+Marketing#
* Reason of Choosing This Dataset
    1) 45000+ instances and 17 attributes which are good scales for testing algorithm efficiency
    2) Real data for business implication
    3) Widely used in other studies
* Evaluation
    1) Are the rules/tree structures constructed being useful/too general?
    2) Training time
    3) Model size
    5) Test set preditction error with different depths
