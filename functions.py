import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
import os


def import_data(folder) -> dict:
    '''
    The function returns the joined dataframes in a format of a dictionary for
    each decade from the given folder.
    '''

    # read filenames from folder containing data
    files = os.listdir(folder)

    r = re.compile("dataset*")
    filtered_files = list(filter(r.match, files))

    # read the csv-s to a dictionary
    df_dict = {}
    for file in filtered_files:
        df_dict[file.split(".")[0]] = (pd.read_csv(f"{folder}/{file}"))
    
    return(df_dict)



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


def gradient_boosting_estimator(X_train, y_train, X_test, y_test, learning_rate, max_depth, est_range):
    '''
    The function runs multiple gradient boosting models with different
    estimator numbers and plots the accuracy of the train and test datasets
    in order to define the optimal number of estimators.
    '''

    acc = []

    for est in tqdm(est_range):
        # creating gradient boosting model
        grad = GradientBoostingClassifier(n_estimators = est, learning_rate = learning_rate , max_depth = max_depth, random_state = 0)
        # fitting model
        grad.fit(X_train, y_train)
        # creating prediction
        y_pred = grad.predict(X_test)
        # evaluating accuracy
        acc.append(accuracy_score(y_test, y_pred))


    # plotting accuracy scores
    fig, ax = plt.subplots(figsize=(8,5))

    ax.plot(est_range, acc)
    # setting design
    ax.set_xlabel('Estimators')
    ax.set_ylabel('Accuracy')
    ax.set_title('Gradient boosting model accuracy on \nestimators', size = 14)


def check_accuracy(y_test, y_pred):
    '''
    This function prints the confusion matrix and the accuracy score of
    the two given arrays.
    '''

    conf_matrix = confusion_matrix(y_test, y_pred)
    acc_score = accuracy_score(y_test, y_pred)

    print(f"Confusion matrix:\n{conf_matrix}\n\nThe accuracy score is {	'{:.4f}'.format(acc_score)}")
