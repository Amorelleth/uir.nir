#!/usr/bin/python3

import os.path
import csv
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV
from mealpy.swarm_based import AO, HGS, SSA, MRFO
from nltk.corpus import stopwords
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot

# bio optimization algorithms
BIO_ALGS = ['AO', 'HGS', 'SSA', 'MRFO']
# ALGS = BIO_ALGS + ['RSCV', 'DEFAULT']
ALGS = BIO_ALGS

# dataset name
DATASETS = ['enron', 'ling_spam', 'spam_assasin']

# size of test part of dataset
SIZE = 0.25


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
    model = clf(problem, epoch=10, pop_size=60)
    model.solve()
    return model.g_best


def main():
    for dataset in DATASETS:
        # load dataset
        df = pd.read_csv(f'./input/{dataset}/messages.csv')
        df = df.fillna(' ')

        X = np.array(df['message'])
        y = np.array(df['label'])

        # split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=SIZE)

        # tokenization
        cv = TfidfVectorizer(stop_words=stopwords.words('english'))
        X_train = cv.fit_transform(X_train)
        X_test = cv.transform(X_test)

        for alg in ALGS:
            # random search
            if alg == 'RSCV':
                tuned_parameters = {
                    'epsilon': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
                    'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
                    'tol': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
                }

                clf = SGDClassifier()
                model = RandomizedSearchCV(
                    clf, tuned_parameters, scoring="roc_auc")
                model.fit(X_train, y_train)
                print(alg, model.best_params_)

            # bio optimization
            elif alg in BIO_ALGS:
                def obj_function(solution):
                    alpha, epsilon, tol = solution
                    clf = SGDClassifier(
                        alpha=alpha, epsilon=epsilon, tol=tol)
                    clf.fit(X_train, y_train)

                    calibrator = CalibratedClassifierCV(clf, cv='prefit')
                    model = calibrator.fit(X_train, y_train)

                    y_score = model.predict_proba(X_test)
                    # print(roc_auc_score(y_test, y_score[:, 1]))

                    # print best params
                    # print(
                    #     f'Alpha={alpha} Epsilon={epsilon} tol={tol} \
                    #         Acc score: {accuracy_score(y_test, y_pred)}')

                    return roc_auc_score(y_test, y_score[:, 1])

                c = resolve_clf(alg)
                best_params_ = test_bio_alg(c, obj_function)
                alpha, epsilon, tol = best_params_[0]

                model = SGDClassifier(
                    alpha=alpha, epsilon=epsilon, tol=tol)
                model.fit(X_train, y_train)

                # print best params
                print(f'{alg} with {dataset} is best with: ',
                      f'{best_params_}'.rjust(29, ' '))

            # default parameters
            else:
                model = SGDClassifier()
                model.fit(X_train, y_train)

            # # generate a no skill prediction (majority class)
            # ns_probs = [0 for _ in range(len(y_test))]

            # calibrator = CalibratedClassifierCV(model, cv='prefit')
            # model = calibrator.fit(X_train, y_train)

            # # predict probabilities
            # lr_probs = model.predict_proba(X_test)
            # # keep probabilities for the positive outcome only
            # lr_probs = lr_probs[:, 1]
            # # calculate scores
            # ns_auc = roc_auc_score(y_test, ns_probs)
            # lr_auc = roc_auc_score(y_test, lr_probs)
            # # summarize scores
            # print('No Skill: ROC AUC=%.3f' % (ns_auc))
            # print('Logistic: ROC AUC=%.3f' % (lr_auc))
            # # calculate roc curves
            # ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
            # lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
            # # plot the roc curve for the model
            # pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
            # pyplot.plot(lr_fpr, lr_tpr, marker='.', label='SGD')
            # # axis labels
            # pyplot.xlabel('False Positive Rate')
            # pyplot.ylabel('True Positive Rate')
            # # show the legend
            # pyplot.legend()
            # # show the plot
            # pyplot.show()

            # precision-recall curve and f1
            from sklearn.datasets import make_classification
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import precision_recall_curve
            from sklearn.metrics import f1_score
            from sklearn.metrics import auc
            from matplotlib import pyplot
            # generate 2 class dataset
            X, y = make_classification(
                n_samples=1000, n_classes=2, random_state=1)
            # split into train/test sets
            trainX, testX, trainy, testy = train_test_split(
                X, y, test_size=0.5, random_state=2)
            # fit a model
            model = LogisticRegression(solver='lbfgs')
            model.fit(trainX, trainy)
            # predict probabilities
            lr_probs = model.predict_proba(testX)
            # keep probabilities for the positive outcome only
            lr_probs = lr_probs[:, 1]
            # predict class values
            yhat = model.predict(testX)
            lr_precision, lr_recall, _ = precision_recall_curve(
                testy, lr_probs)
            lr_f1, lr_auc = f1_score(testy, yhat), auc(lr_recall, lr_precision)
            # summarize scores
            print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
            # plot the precision-recall curves
            no_skill = len(testy[testy == 1]) / len(testy)
            pyplot.plot([0, 1], [no_skill, no_skill],
                        linestyle='--', label='No Skill')
            pyplot.plot(lr_recall, lr_precision, marker='.', label='Logistic')
            # axis labels
            pyplot.xlabel('Recall')
            pyplot.ylabel('Precision')
            # show the legend
            pyplot.legend()
            # show the plot
            pyplot.show()


if __name__ == '__main__':
    main()


# HGS 0.0001, 0.0001, 0.0001
