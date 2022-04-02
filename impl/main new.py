#!/usr/bin/python3

import os.path
import csv
import numpy as np
import pandas as pd

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
ALGS = BIO_ALGS + ['RSCV', 'DEFAULT']

# log fields
FIELDS = ['Алгоритм', 'Данные', 'Accuracy',
          'Precision', 'Recall', 'F1-score']

# iterations number
ITERS = 100

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


def test_bio_alg(clf, obj_function):
    problem = {
        'obj_func': obj_function,
        'lb': [0.0001, 0.0001, 0.0001],
        'ub': [1000, 1000, 1000],
        'minmax': 'max',
        'verbose': True,
    }
    model = clf(problem, epoch=10, pop_size=40)
    model.solve()
    return model.g_best


def main():
    for alg in ALGS:

        alpha = 0.0001
        epsilon = 0.0001
        tol = 0.0001

        with open(r'log.csv', 'a') as f:
            writer = csv.writer(f)

            if not os.path.isfile('log.csv') or \
                    os.path.getsize('log.csv') == 0:
                writer.writerow(FIELDS)

            # load dataset
            df = pd.read_csv(f'./input/{DATASET}/messages.csv')
            df = df.fillna(' ')

            X = np.array(df['message'])
            y = np.array(df['label'])

            sum_acc = []
            sum_prec = []
            sum_recall = []
            sum_f1 = []
            sum_t = []

            for iter in range(0, ITERS):
                print("Iter", iter)

                t = 0
                y_pred = []

                # split dataset
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=SIZE)

                # tokenization
                cv = TfidfVectorizer(stop_words=stopwords.words('english'))
                X_train = cv.fit_transform(X_train)
                X_test = cv.transform(X_test)

                # random search
                if alg == 'RSCV':
                    tuned_parameters = {
                        'epsilon': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
                        'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
                        'tol': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
                    }

                    clf = SGDClassifier()
                    model = RandomizedSearchCV(
                        clf, tuned_parameters, scoring="f1")
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                # bio optimization
                elif alg in BIO_ALGS:
                    def obj_function(solution):
                        alpha, epsilon, tol = solution
                        clf = SGDClassifier(
                            alpha=alpha, epsilon=epsilon, tol=tol)
                        clf.fit(X_train, y_train)
                        y_pred = clf.predict(X_test)

                        # print best params
                        # print(
                        #     f'Alpha={alpha} Epsilon={epsilon} tol={tol} \
                        #         Acc score: {accuracy_score(y_test, y_pred)}')

                        return f1_score(y_test, y_pred)

                    c = resolve_clf(alg)
                    best_params_ = test_bio_alg(c, obj_function)
                    alpha, epsilon, tol = best_params_[0]

                    clf = SGDClassifier(
                        alpha=alpha, epsilon=epsilon, tol=tol)
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)

                    # print best params
                    # print(f'Best with: ', f'{best_params_}'.rjust(29, ' '))

                # default parameters
                else:
                    clf = SGDClassifier()
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)

                sum_acc += [acc]
                sum_prec += [prec]
                sum_recall += [recall]
                sum_f1 += [f1]

            # calc average metrics
            avg_acc = statistics.median(sum_acc)
            avg_prec = statistics.median(sum_prec)
            avg_recall = statistics.median(sum_recall)
            avg_f1 = statistics.median(sum_f1)

            # write result row
            writer.writerow([alg, DATASET, format_precent(avg_acc),
                            format_precent(avg_prec), format_precent(
                                avg_recall),
                            format_precent(avg_f1)])

        f.close()


if __name__ == '__main__':
    main()
