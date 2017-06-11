import time
import matplotlib.pyplot as plt
import numpy as np
import sklearn.svm as svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

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

    """
    Pretty good accuracy score, but takes a while to train because of the large
    number of data ~50000 and features ~ 600. SGD on the other hand is instantaneous
    because it's designed for large scale data
    """

def dTree(X, y, X_train, y_train, X_test, y_test):
    """
    This function will tune, train, and time the training of a Decision
        tree. We will use the tuned classifier to evaluate it on a
        variety of performance metrics
    Parameters
    ----------
        X, y -- The original dataset and labels
        X_train, y_train -- Training dataset
        X_test, y_test -- Testing dataset
    Return
    ------
        clf -- Hyperparameter tuned classifier
    """
    # 1. Decision Tree
    # Finding best max depth to overcome overfitting
    # depth = np.arange(1,30)
    # trainError = []
    # testError = []
    # for d in depth:
    #     t1, t2 = error(DecisionTreeClassifier(max_depth=d), X, y)
    #     trainError.append(t1)
    #     testError.append(t2)
    #     print "Finished calculations for depth " + str(d)
    # plt.plot(depth, trainError, "r-", label="Tree Training Error")
    # plt.plot(depth, testError, "b-", label="Tree Test Error")
    # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    # plt.xlabel("Max Depth")
    # plt.ylabel("Error")
    # plt.show()

    # Best Max Depth = 20 (look at dectree.png)
    clf = DecisionTreeClassifier(max_depth=20)

    # Print time elapsed for training
    start = time.time()
    clf.fit(X_train, y_train)
    print "DTree Time: " + str(time.time() - start)

    # print "Decision Tree Accuracy Score: " + str(clf.score(X_test, y_test))
    return clf

def svmlin(X_train, y_train, X_test, y_test):
    """
    This function will tune, train, and time the training of a Support Vector
        Machine with linear kernel. We will use the tuned classifier to
        evaluate it on a variety of performance metrics
    Parameters
    ----------
        X, y -- The original dataset and labels
        X_train, y_train -- Training dataset
        X_test, y_test -- Testing dataset
    Return
    ------
        clf -- Hyperparameter tuned classifier
    """
    # Tuning hyperparameters -- C
    # clf = svm.SVC(kernel='linear')
    # parameters = {'C':[1, 10, 100]}
    # gs = GridSearchCV(clf, parameters)
    # gs.fit(X_train, y_train)
    # print "Optimal hyperparameters: " + str(gs.best_params_)

    # Training classifier with best parameter
    # clf_lin = svm.SVC(kernel='linear', C=gs.best_params_['C'])
    clf_lin = svm.SVC(kernel='linear', C=10)
    # Print time elapsed for training
    start = time.time()
    clf_lin.fit(X_train, y_train)
    print(time.time() - start)

    # print "Linear Kernel SVM Accuracy Score: " + str(clf_lin.score(X_test, y_test))
    return clf_lin

def svmrbf(X_train, y_train, X_test, y_test):
    """
    This function will tune, train, and time the training of a Support Vector
        Machine with rbf kernel. We will use the tuned classifier to
        evaluate it on a variety of performance metrics
    Parameters
    ----------
        X, y -- The original dataset and labels
        X_train, y_train -- Training dataset
        X_test, y_test -- Testing dataset
    Return
    ------
        clf -- Hyperparameter tuned classifier
    """
    # Tuning hyperparameters -- C, gamma
    clf = svm.SVC(kernel='rbf')
    parameters = {'C':[.1, 1, 10, 100], 'gamma':[.01, .1, 1, 10]}
    gs = GridSearchCV(clf, parameters)
    gs.fit(X_train, y_train)
    print "Optimal hyperparameters: " + str(gs.best_params_)

    # Training the classifier
    clf_rbf = svm.SVC(kernel='rbf', C=gs.best_params_['C'], gamma=gs.best_params_['gamma'])

    # Print time elapsed for training
    start = time.time()
    clf_rbf.fit(X_train, y_train)
    print(time.time() - start)

    # print "RBF Kernel SVM Accuracy Score: " + str(clf_rbf.score(X_test, y_test))
    return clf_rbf

def sgd(X_train, y_train, X_test, y_test):
    """
    This function will tune, train, and time the training of a Stochastic
        Gradient Descent classifier. We will use the tuned classifier to
        evaluate it on a variety of performance metrics
    Parameters
    ----------
        X, y -- The original dataset and labels
        X_train, y_train -- Training dataset
        X_test, y_test -- Testing dataset
    Return
    ------
        clf -- Hyperparameter tuned classifier
    """
    # 2. SGD Classifer using Linear SVM
    # Two hyperparameters to tune: n_iter and alpha (are there more?)
    # Preprocess Data http://scikit-learn.org/stable/modules/sgd.html#tips-on-practical-use
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_trainPre = scaler.transform(X_train)
    X_testPre = scaler.transform(X_test)

    # # Tuning hyperparameters -- alpha, niter
    # sgd = linear_model.SGDClassifier()
    # parameters = {'alpha':[0.0001, 0.001, 0.01, 0.1], 'n_iter':[5,10,15,20,25,30]}
    # gs = GridSearchCV(sgd, parameters)
    #
    # gs.fit(X_trainPre, y_train)
    # print "Optimal hyperparameters: " + str(gs.best_params_) # 0.001, 25

    # Training the hyperparameter-tuned model
    sgd = linear_model.SGDClassifier(alpha=0.001, n_iter=25) #from above
    # Print time elapsed for training
    start = time.time()
    sgd.fit(X_trainPre, y_train)
    print "SGD Time: " + str(time.time() - start)

    # print "SGD Accuracy Score: " + str(sgd.score(X_testPre, y_test)) # 83%
    return sgd, X_testPre

def logreg(X_train, y_train, X_test, y_test):
    """
    This function will tune, train, and time the training of a Logistic
        Regression classifier. We will use the tuned classifier to
        evaluate it on a variety of performance metrics
    Parameters
    ----------
        X, y -- The original dataset and labels
        X_train, y_train -- Training dataset
        X_test, y_test -- Testing dataset
    Return
    ------
        clf -- Hyperparameter tuned classifier
    """
    # Tuning single hyperparameter -- C
    # parameters = {'C':[100, 200, 500, 1000]}
    # gs = GridSearchCV(linear_model.LogisticRegression(), parameters)
    # gs.fit(X_train, y_train)
    # print "Optimal hyperparameters: " + str(gs.best_params_)

    log = linear_model.LogisticRegression(C=1000)

    # Print time elapsed for training
    start = time.time()
    log.fit(X_train, y_train)
    print "Log Time: " + str(time.time() - start)

    # print "Logistic Regression Accuracy Score: " + str(log.score(X_test, y_test))
    return log

def gradboost(X_train, y_train, X_test, y_test):
    """
    This function will tune, train, and time the training of a gradient
        boosted classifier. We will use the tuned classifier to
        evaluate it on a variety of performance metrics
    Parameters
    ----------
        X, y -- The original dataset and labels
        X_train, y_train -- Training dataset
        X_test, y_test -- Testing dataset
    Return
    ------
        clf -- Hyperparameter tuned classifier
    """
    # Tuning hyperparameters
    # parameters = {'n_estimators':[15,25,50]}
    # gs = GridSearchCV(GradientBoostingClassifier(), parameters)
    # gs.fit(X_train, y_train)
    # print "Optimal hyperparameters: " + str(gs.best_params_)

    grd = GradientBoostingClassifier(max_depth=20, n_estimators=100, loss='deviance', learning_rate=1)

    # Print time elapsed for training
    start = time.time()
    grd.fit(X_train, y_train)
    print "GRD Time: " + str(time.time() - start)
    grd.score(X_test, y_test)

    # print "Gradient Boosted Accuracy Score: " + str(grd.score(X_test, y_test))
    return grd

def performance(y_true, y_pred):
    """
    Parameters
    ----------
        y_true -- numpy array (n,), known labels
        y_pred -- numpy array (n,), our predictions
        metric -- the performance measure we want to see: currently handles
            accuracy, f1-score, precision, sensitivity, and specificity
    Return
    ------
        score -- all metrics
    """
    mets = {}
    mets["accuracy"] = metrics.accuracy_score(y_true, y_pred)
    mets["precision"] = metrics.precision_score(y_true, y_pred)
    conf = metrics.confusion_matrix(y_true, y_pred, labels=[1,0])
    mets["sensitivity"] = float(conf[0,0]) / np.sum(conf[0,:])
    mets["specificity"] = float(conf[1,1]) / np.sum(conf[1,:])
    return mets


def main():
    # x is 50,000 * 622 i.e. 50,000 data points and 622 features
    X = np.loadtxt("train_data_final2 copy.csv", delimiter=",", usecols=range(0,622))
    # y is 50,000 * 1. Labels are either 1 or 0
    y = np.loadtxt("train_data_final2 copy.csv", delimiter=",", usecols=(622,))

    print "Raw data initialized..."

    kf = KFold(n_splits=5)  # n = 5

    logResult = []
    sgdResult = []
    dtreeResult = []
    grdResult = []

    metrics = ["accuracy", "precision", "sensitivity", "specificity"]
    n = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # svms too slow for this large dataset
        # clf = svmlin(X_train, y_train, X_test, y_test)
        # print np.unique(clf.predict(X_test))

        #logreg will predict[0,1] // ~ 169s of training time // accuracy ~ 0.84
        #hyperparameters: C = 1000
        clf_log = logreg(X_train, y_train, X_test, y_test)
        logResult.append(clf_log.score(X_test, y_test))

        print performance(y_test,clf_log.predict(X_test))
        print "----------------------------"

        #sgd will predict[0,1] // ~ 1.28s of training time // accuracy ~ 0.83
        #hyperparameters: alpha = 0.001, n_iters = 25
        clf_sgd, X_testPre = sgd(X_train, y_train, X_test, y_test)
        sgdResult.append(clf_sgd.score(X_testPre, y_test))

        print performance(y_test,clf_sgd.predict(X_testPre))
        print "----------------------------"

        #dtree will predict [0,1] // ~ 44s of training time // accuracy ~ 0.96
        #hyperparameters: depth = 20
        clf_dtree = dTree(X, y, X_train, y_train, X_test, y_test)
        dtreeResult.append(clf_dtree.score(X_test, y_test))

        print performance(y_test,clf_dtree.predict(X_test))
        print "----------------------------"

        #gradient boosting will predict [0,1] // ~ 770s of training time // accuracy ~ 0.97
        #hyperparameters: loss = deviance, learning_rate = 1, n_estimators = 100
        clf_grd = gradboost(X_train, y_train, X_test, y_test)
        grdResult.append(clf_grd.score(X_test, y_test))

        print performance(y_test,clf_grd.predict(X_test))
        print "----------------------------"

        print "Finished fold " + str(n)
        n = n + 1

    print "========= Training Results (Accuracy Scores across n-Folds) ========="
    print "Log Reg: " + str(logResult) + " avg: " + str(sum(logResult) / float(len(logResult)))
    print "SGD: " + str(sgdResult) + " avg: " + str(sum(sgdResult) / float(len(sgdResult)))
    print "DTree: " + str(dtreeResult) + " avg: " + str(sum(dtreeResult) / float(len(dtreeResult)))
    print "GRD: " + str(grdResult) + " avg: " + str(sum(grdResult) / float(len(grdResult)))

if __name__ == "__main__":
    main()
