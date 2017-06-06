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
    print(time.time() - start)
    
    print "Decision Tree Accuracy Score: " + str(clf.score(X_test, y_test))
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
    
    print "Linear Kernel SVM Accuracy Score: " + str(clf_lin.score(X_test, y_test))
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
    
    print "RBF Kernel SVM Accuracy Score: " + str(clf_rbf.score(X_test, y_test))
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
    print(time.time() - start)
    
    print "SGD Accuracy Score: " + str(sgd.score(X_testPre, y_test)) # 83%
    return sgd

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
    parameters = {'C':[100, 200, 500]}
    gs = GridSearchCV(linear_model.LogisticRegression(), parameters)
    gs.fit(X_train, y_train)
    print "Optimal hyperparameters: " + str(gs.best_params_)

    log = linear_model.LogisticRegression(C=gs.best_params_['C'])

    # Print time elapsed for training
    start = time.time()
    log.fit(X_train, y_train)
    print(time.time() - start)

    print "Logistic Regression Accuracy Score: " + str(log.score(X_test, y_test))
    # Around 83% too or slightly higher, run it again to just make sure
    return log

def performance(y_true, y_pred, metric="accuracy"):
    """
    Parameters
    ----------
        y_true -- numpy array (n,), known labels
        y_pred -- numpy array (n,), our predictions
        metric -- the performance measure we want to see: currently handles
            accuracy, f1-score, precision, sensitivity, and specificity
    Return
    ------
        score -- (float) the metric performance score
    """
    # map predictions to binary values
    y_label = np.sign(y_pred)
    #y_label[y_label==0] = 1

    mets = {}
    mets["accuracy"] = metrics.accuracy_score(y_true, y_label)
    mets["f1-score"] = metrics.f1_score(y_true, y_label)
    mets["precision"] = metrics.precision_score(y_true, y_label)
    conf = metrics.confusion_matrix(y_true, y_label, labels=[1,-1])
    mets["sensitivity"] = float(conf[0,0]) / np.sum(conf[0,:]) 
    mets["specificity"] = float(conf[1,1]) / np.sum(conf[1,:])
    return mets[metric]

def main():
    # x is 50,000 * 622 i.e. 50,000 data points and 622 features
    X = np.loadtxt("train_data_final2 copy.csv", delimiter=",", usecols=range(0,622))
    # y is 50,000 * 1. Labels are either 1 or 0
    y = np.loadtxt("train_data_final2 copy.csv", delimiter=",", usecols=(622,))

    print "Raw data initialized..."
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # splits 20% into test data

    #svms too slow
    #clf = svmlin(X_train, y_train, X_test, y_test)   
    #print np.unique(clf.predict(X_test))

    #logreg will predict[0,1] // ~ 169s of training time // accuracy ~ 0.84
    clf = logreg(X_train, y_train, X_test, y_test)

    #sgd will predict[0,1] // ~ 1.28s of training time // accuracy ~ 0.83
    #hyperparameters: alpha = 0.001, n_iters = 25
    #clf_sgd = sgd(X_train, y_train, X_test, y_test)  

    #dtree will predict [0,1] // ~44s of training time // accuracy ~ 0.96
    #hyperparameters: depth = 20
    #clf_dtree = dTree(X, y, X_train, y_train, X_test, y_test)

    # Gradient Boosting

    #grd = GradientBoostingClassifier(max_depth=20, n_estimators=10)
    #grd.fit(X_train, y_train)
    #grd.score(X_test, y_test)

if __name__ == "__main__":
    main()
