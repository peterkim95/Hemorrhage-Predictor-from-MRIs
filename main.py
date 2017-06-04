import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model

def error(clf, X, y, ntrials=1, test_size=0.2) :
    """
    Computes the classifier error over a random split of the data,
    averaged over ntrials runs.
    """
    trainErrorSum = 0
    testErrorSum = 0
    for x in range(0,ntrials):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=x)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_train)
        trainErrorSum += 1 - metrics.accuracy_score(y_train, y_pred, normalize=True)
        y_pred = clf.predict(X_test)
        testErrorSum += 1 - metrics.accuracy_score(y_test, y_pred, normalize=True)

    train_error = trainErrorSum / ntrials
    test_error = testErrorSum  / ntrials

    return train_error, test_error

def main():
    # x is 50,000 * 622 i.e. 50,000 data points and 622 features
    X = np.loadtxt("train_data_final2 copy.csv", delimiter=",", usecols=range(0,622))
    # y is 50,000 * 1. Labels are either 1 or 0
    y = np.loadtxt("train_data_final2 copy.csv", delimiter=",", usecols=(622,))

    print "Raw data initialized..."
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # splits 20% into test data

    # 1. Decision Tree
    # Finding best max depth to overcome overfitting
    depth = np.arange(1,30)
    trainError = []
    testError = []

    for d in depth:
        t1, t2 = error(DecisionTreeClassifier(max_depth=d), X, y)
        trainError.append(t1)
        testError.append(t2)
        print "Finished calculations for depth " + str(d)

    plt.plot(depth, trainError, "r-", label="Tree Training Error")
    plt.plot(depth, testError, "b-", label="Tree Test Error")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.xlabel("Max Depth")
    plt.ylabel("Error")
    plt.show()

    # Best Max Depth = 20 (look at dectree.png)
    clf = DecisionTreeClassifier(max_depth=20)
    clf.fit(X_train, y_train)
    print "Decision Tree Accuracy Score: " + str(clf.score(X_test, y_test))

    # Pretty good accuracy score, but takes a while to train because of the large number of data ~50000 and features ~ 600. SGD on the other hand is instantaneous because it's designed for large scale data

    # 2. SGD Classifer using Linear SVM
    # Two hyperparameters to tune: n_iter and alpha (are there more?)

    # Preprocess Data http://scikit-learn.org/stable/modules/sgd.html#tips-on-practical-use
    # scaler = StandardScaler()
    # scaler.fit(X_train)
    # X_trainPre = scaler.transform(X_train)
    # X_testPre = scaler.transform(X_test)
    #
    # sgd = linear_model.SGDClassifier()
    #
    # # Find optimal hyperparameters through gridsearch
    # parameters = {'alpha':[0.0001, 0.001, 0.01, 0.1], 'n_iter':[5,10,15,20,25,30]}
    # gs = GridSearchCV(sgd, parameters)
    # gs.fit(X_trainPre, y_train)
    # print "Optimal hyperparameters: " + str(gs.best_params_) # 0.001, 25
    #
    # # Use them for final training and accuracy test
    # sgd = linear_model.SGDClassifier(alpha=gs.best_params_['alpha'], n_iter=gs.best_params_['n_iter'])
    # sgd.fit(X_trainPre, y_train)
    #
    # print "SGD Accuracy Score: " + str(sgd.score(X_testPre, y_test)) # 83%

    # 3. Logistic Regression
    # One hyperparameter - C

    parameters = {'C':[0.0001,0.001,0.01,0.1,1,10,100,1000]}
    gs = GridSearchCV(linear_model.LogisticRegression(), parameters)
    gs.fit(X_train, y_train)
    print "Optimal hyperparameters: " + str(gs.best_params_)

    log = linear_model.LogisticRegression(C=gs.best_params_['C'])
    log.fit(X_train, y_train)

    print "Logistic Regression Accuracy Score: " + str(log.score(X_test, y_test))
    # Around 83% too or slightly higher, run it again to just make sure

if __name__ == "__main__":
    main()
