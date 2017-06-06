# Introduction
We were tasked with designing a good predictor for whether someone had a hemorrhage based on MRIs. The data consist of rows of length 622. We have 50000 of these rows, and each row represents the pixels found in a MRI image. Each row/image is paired with a boolean, 0 or 1, denoting whether the image shows a non-hemorrhaged or a hemorrhaged brain. We did an 80/20 split for the training/testing data. To tackle this problem, we immediately set out to train a variety of different models to solve this classification problem.

# Hyperparameter Tuning
Naturally, some of these classifiers come with hyperparameters that needed to be tuned before being used to test. To tune these hyperparameters, we made use of k-fold cross validation. We split the data into several folds, and picked the combination of hyperparameters that achieve the highest average accuracy score across all folds (as accuracy is the main performance metric that we care about). An easy example of this is how we chose the maximum depth hyperparameter of our decision tree classifier. Below is the cross validation graph.

![alt test](https://github.com/peterkim95/Hemorrhage-Predictor-from-MRIs/blob/master/dectree.png)

What we really seek is for a good test accuracy. We see that the test accuracy stagnates at around a maxdepth of 20 while the training error continues to decrease. That is a sign of overfitting. This is why we settled for a maxdepth of 20. We used this technique to tune the hyperparameters for all of our classifiers.

# Performance Metrics
We did not want to solely rely on test accuracy as a performance metric. We also wanted to test each of our classifiers based on several other performance metrics. These include precision, sensitivity, and specificity. Precision tells us how well our model predicts positives. A precision score of 1 means that all of the model's positive predictions are true positives while a precision score of 0 means all of its positive predictions are false positives. Sensitivity tells us the probability of detection. More specifically, it tells us what the percentage of hemorrhages the model correctly predicted. Finally, specificity is the foil to sensitivity. Specificity tells us what percentage of the non-hemorrhages were correctly predicted.

# Our First Model -- Kernelized SVM
We know that theoretically, SVM is the most robust and widely used classification model. So we decided to try this right off the bat. We wanted to try two of the most popular kernels -- RBF and linear, but before we even finished the cross-validation stage of building our model, we noticed that it took way too long to train an SVM given the size of our dataset. Thus, we sought to try other methods.

# Faster -- Logistic Regression
Perhaps we decided to go for too complex of a classifier. We decided to try a more basic classifier -- logistic regression. Below you can find a table of this model's attributes.

| Training Time | Accuracy | Precision | Sensitivity | Specificity |
|:-------------:|:--------:|:---------:|:-----------:|:-----------:|
| ~169s | 0 | 0 | 0 | 0 |

# Much Faster -- Stochastic Gradient Descent
Still, we decided that approximately three minutes is too slow for such a classifier. While logistic regression is possible implmeneted using batch or minibatch gradient descent, we decided to make the iterative step as fast as possible by utilizing stochastic gradient descent. Stochastic gradient descent calculates a gradient on a single training example at a time rather than operating on a subset or even the entire training dataset. No wonder it can finish training in as fast as a second!

| Training Time | Accuracy | Precision | Sensitivity | Specificity |
|:-------------:|:--------:|:---------:|:-----------:|:-----------:|
| ~1s | 0 | 0 | 0 | 0 |

# More Accurate -- Decision Trees
Although SGD is absurdly fast, 83% accuracy is far too low. Surprisingly, decision trees was a model that yielded high accuracy scores in a reasonable amount of training time.

| Training Time | Accuracy | Precision | Sensitivity | Specificity |
|:-------------:|:--------:|:---------:|:-----------:|:-----------:|
| ~44s | 0 | 0 | 0 | 0 |

# Even More Accurate -- Gradient Boosting (Ensemble)
If a simple learner like a decision tree can achieve such a high accuracy score, why don't we try to further improve the accuracy by using an ensemble of such learners?

| Training Time | Accuracy | Precision | Sensitivity | Specificity |
|:-------------:|:--------:|:---------:|:-----------:|:-----------:|
| ~169s | 0 | 0 | 0 | 0 |