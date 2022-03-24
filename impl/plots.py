#!/usr/bin/python3

import os.path
import csv
import numpy as np
import pandas as pd
import time

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from mealpy.swarm_based import AO, HGS, SSA, MRFO
from nltk.corpus import stopwords
import statistics

# bio optimization algorithms
BIO_ALGS = ['AO', 'HGS', 'SSA', 'MRFO']
# ALGS = BIO_ALGS + ['RSCV', 'DEFAULT']
ALGS = ['AO']

# log fields
FIELDS = ['alg', 'data', 'test-size', 'accuracy',
          'precision', 'recall', 'f1-score']

# dataset name
DATASET = 'ling_spam'

# size of test part of dataset
SIZE = 0.25


def format_precent(val):
    return "{:.2f}%".format(val * 100)


def resolve_clf(alg):
    if alg == 'AO':
        return AO.OriginalAO
    elif alg == 'HGS':
        return HGS.OriginalHGS
    elif alg == 'SSA':
        return SSA.OriginalSSA
    elif alg == 'MRFO':
        return MRFO.BaseMRFO


def main():
    df = pd.read_csv(f'./input/{DATASET}/messages.csv')
    df = df.fillna(' ')

    X = np.array(df['message'])
    y = np.array(df['label'])

    cv = TfidfVectorizer(stop_words=stopwords.words('english'))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=SIZE)

    X_train = cv.fit_transform(X_train)
    X_test = cv.transform(X_test)

    def obj_function(solution):
        alpha, epsilon, tol = solution
        clf = SGDClassifier(
            alpha=alpha, epsilon=epsilon, tol=tol)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return f1_score(y_test, y_pred)

    problem = {
        'obj_func': obj_function,
        'lb': [0.0001, 0.0001, 0.0001],
        'ub': [1000, 1000, 1000],
        'minmax': 'max',
        'verbose': True,
    }

    model = HGS.OriginalHGS(problem, epoch=10, pop_size=50)
    # model = MRFO.BaseMRFO(problem, epoch=10, pop_size=50)
    model.solve()

    model.history.save_exploration_exploitation_chart()
    model.history.save_diversity_chart()
    model.history.save_global_best_fitness_chart()
    model.history.save_global_objectives_chart()
    model.history.save_runtime_chart()
    model.history.save_trajectory_chart()


if __name__ == '__main__':
    main()
