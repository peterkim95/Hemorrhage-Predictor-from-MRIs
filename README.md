# Hemorrhage Prediction from MRIs

Our code consists of a main function, all the classifier training functions, and a performance evaluation function.

Each of the classifier training function takes in X,y train and test sets and returns the final trained classifiers. It first uses cross validation to help choose the hyperparameters (we commented those sections of code out because we already tuned the hyperparameters) then uses those hyperparameters to train the classifier.

The performance measuring function takes in y_test (the actual test labels), the predicted y labels, and the performance metric that we want to evaluate. It will return the float value of the performance metric.

The main function will first split the raw data into n-folds (n=5 - can change to higher if desired), and train each of the classifiers for every single fold. For each fold, the classifier's metrics (accuracy, specificity, precision, sensitivity) and time taken will be printed onto the console.

Simply place the training data file (named "train_data_final2.csv"), which can be obtained from Professor Fabien Scalzo, in the same directory as main.py. Then, you can run our code by running main.py
