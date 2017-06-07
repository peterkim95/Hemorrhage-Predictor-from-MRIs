# Hemorrhage-Predictor-from-MRIs

Our code consists of a main function, all the classifier training functions, and a performance evaluation function.

Each of the classifier training function takes in X,y train and test sets and returns the final trained classifiers. It first uses cross validation to help choose the hyperparameters (we commented those sections of code out because we already tuned the hyperparameters) then uses those hyperparameters to train the classifier.

The performance measuring function takes in y_test (the actual test labels), the predicted y labels, and the performance metric that we want to evaluate. It will return the float value of the performance metric.

The main function will call each of the classifier training functions and save the returned/trained classifiers in a list. When we have gathered all the trained classifiers, it'll iterate through all of them and calculate all the performance metrics for it.