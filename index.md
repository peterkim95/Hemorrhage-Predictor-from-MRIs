# Introduction
We were tasked with designing a good predictor for whether someone had a hemorrhage based on MRIs. The data consist of rows of length 622. We have 50000 of these rows, and each row represents an MRI image. Each row/image is paired with a boolean, 0 or 1, denoting whether the image shows a non-hemorrhaged or a hemorrhaged brain. We did an 80/20 split for the training/testing data. To tackle this problem, we immediately set out to train a variety of different models to solve this classification problem.

# Methods
Naturally, some of these classifiers come with hyperparameters that needed to be tuned before being used to test. To tune these hyperparameters, we made use of k-fold cross validation. We split the data into several folds, and picked the combination of hyperparameters that achieve the highest average accuracy score across all folds (as accuracy is the main performance metric that we care about). An easy example of this is how we chose the maximum depth hyperparameter of our decision tree classifier. Below is the cross validation graph.

![Image](/dectree.png)

What we really seek is for a good test accuracy. We see that the test accuracy stagnates at around a max depth of 20 while the training error continues to decrease. That is a sign of overfitting. This is why we settled for a max depth of 20. We used this technique to tune the hyperparameters for all of our classifiers.

Otherwise, we employed the GridSearchCV method in the scikit-learn API to determine the best hyperparameters for each classier. Observe the following example for our SGD classifier

```
# Tuning hyperparameters -- alpha, n_iter
sgd = linear_model.SGDClassifier()
parameters = {'alpha':[0.0001, 0.001, 0.01, 0.1], 'n_iter':[5,10,15,20,25,30]}
gs = GridSearchCV(sgd, parameters)
# we start the exhaustive grid search using the given training data
gs.fit(X_trainPre, y_train)
print "Optimal hyperparameters: " + str(gs.best_params_)
```

We used these techniques to train and tune all of the the models that we used to tackle this problem as discussed below:

We first tried to train with SVM. We know that theoretically, SVM is the most robust and widely used classification model. So we decided to try this right off the bat. We wanted to try two of the most popular kernels -- RBF and linear, but before we even finished the cross-validation stage of building our model, we noticed that it took way too long to train an SVM given the size of our dataset. Thus, we sought to try other methods.

But perhaps we decided to go for too complex of a classifier. Next, we decided to try a more basic classifier -- logistic regression. Below you can find a table of this model's attributes.

Still, we decided that approximately three minutes is too slow for such a classifier with subpar accuracy. While logistic regressions can possibly be implemented using batch or mini-batch gradient descent, we decided to make the iterative step as fast as possible by utilizing stochastic gradient descent (SGD) using Linear SVM. Stochastic gradient descent calculates a gradient on a single training example at a time rather than operating on a subset or even the entire training dataset. No wonder it can finish training in as fast as a second!

Although SGD is absurdly fast, 83% accuracy is far too low. We want to find a classifier that can achieve a high accuracy as well as train in a reasonable amount of time. Surprisingly, decision trees was a model that yielded high accuracy scores in a reasonable amount of training time.

Finally, we decided to try to achieve the highest accuracy possible with less regard to training time. If a simple learner like a decision tree can achieve such a high accuracy score, why don't we try to further improve the accuracy by using an ensemble of such learners? In this section we decided to imagine a scenario in which we had a much greater amount of computing power -- how could we create the most accurate classifier? The answer is an ensemble of 500 weak learners. More learners would further improve the accuracy, but with diminishing returns. The performance statistics are the best as clearly observable from the table below.

# Evaluation
We did not want to solely rely on test accuracy as a performance metric. We also wanted to test each of our classifiers based on several other performance metrics. These include precision, sensitivity, and specificity. Precision tells us how well our model predicts positives. A precision score of 1 means that all of the model's positive predictions are true positives while a precision score of 0 means all of its positive predictions are false positives. Sensitivity tells us the probability of detection. More specifically, it tells us what the percentage of hemorrhages the model correctly predicted. Finally, specificity is the foil to sensitivity. Specificity tells us what percentage of the non-hemorrhages were correctly predicted.

# Results
Logistic Regression
| Training Time | Accuracy | Precision | Sensitivity | Specificity |
|:-------------:|:--------:|:---------:|:-----------:|:-----------:|
| ~232s | .85 | .83 | .88 | .81 |

Stochastic Gradient Descent
| Training Time | Accuracy | Precision | Sensitivity | Specificity |
|:-------------:|:--------:|:---------:|:-----------:|:-----------:|
| ~1s | .83 | .81 | .84 | .80 |

Decision Trees
| Training Time | Accuracy | Precision | Sensitivity | Specificity |
|:-------------:|:--------:|:---------:|:-----------:|:-----------:|
| ~44s | .96 | .94 | .98 | .94 |

Gradient Boosting (Ensemble)
| Training Time | Accuracy | Precision | Sensitivity | Specificity |
|:-------------:|:--------:|:---------:|:-----------:|:-----------:|
| ~2316s | .98 | .97 | .99 | .97 |

# Discussion
Overall, our results can be summarized by the following table:

| Classifier | Training Time | Test Accuracy |
|:----------:|:-------------:|:-------------:|
| SVM | too slow | unknown (probably high) |
| Logistic Regression | ~232s | .85 |
| SGD (w/ Linear SVM) | ~1s | .83 |
| Decision Tree | ~44s | .96 |
| Gradient Boosting (w/ 500 learners) | ~2316s | .98 |

Clearly, if one is attempting to attain the best accuracy, an ensemble learning method such as Gradient Boosting will suffice. However, one must take account of the exponentially longer training time to use Gradient Boosting. Instead, we recommend using a decision tree of max depth = 20 to train the data, as it only takes a split fraction of the time and yet gives a high accuracy of 96%. If one is searching for the fastest solution, then certainly SGD is the method to employ as the training time is almost instantaneous.
