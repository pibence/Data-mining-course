import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def random_forest_estimator(X_train, y_train, X_test, y_test, max_depth):
    '''
    The function runs multiple random forest models with different estimator
    numbers and plots the accuracy of the train and test datasets in order to define
    the optimal number of estimators.
    '''
    

    # defining range of estimators based on previous tries on the dataset
    estimators = range(5, 25)

    train_acc = np.zeros(len(estimators))
    test_acc = np.zeros(len(estimators))

    for i, est in enumerate(estimators):
        # creating random forest model
        clf = RandomForestClassifier(n_estimators = est, random_state = 0, max_depth = max_depth)
        # fitting model
        clf = clf.fit(X_train, y_train)
        # predicting on train and test set
        y_predtrain = clf.predict(X_train)
        y_predtest = clf.predict(X_test)
        # calculating accuracy scores
        train_acc[i] = accuracy_score(y_train, y_predtrain)
        test_acc[i] = accuracy_score(y_test, y_predtest)

    # plotting accuracy scores
    fig, ax = plt.subplots(figsize=(8,5))

    ax.plot(estimators, train_acc, 'ro-', estimators, test_acc, 'bv--')
    # setting design
    ax.set_xlabel('Estimators')
    ax.set_ylabel('Accuracy')
    ax.legend(['Training Accuracy','Test Accuracy'])
    ax.set_title("Accuracy of random forest model", size = 14)


    def gradient_boosting_estimator(X_train, y_train, X_test, y_test, learnig_rate, max_depth, est_rangethe):
        '''
        The function runs multiple gradient boosting models with different
        estimator numbers and plots the accuracy of the train and test datasets
        in order to define the optimal number of estimators.
        '''

