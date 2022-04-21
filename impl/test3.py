from contextlib import redirect_stdout
from sklearn.model_selection import StratifiedKFold
from mealpy.swarm_based import AO, HGS, SSA, MRFO, HHO
from matplotlib import pyplot
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, auc
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV
import numpy as np
import pandas as pd
from numpy import mean
from numpy import std

enron = pd.read_csv(f'./input/enron/messages.csv').fillna(' ')
X_enron = np.array(enron['message'])
y_enron = np.array(enron['label'])

ling_spam = pd.read_csv(f'./input/ling_spam_copy/messages.csv').fillna(' ')
X_ling_spam = np.array(ling_spam['message'])
y_ling_spam = np.array(ling_spam['label'])

spam_assasin = pd.read_csv(
    f'./input/spam_assasin_copy/messages.csv').fillna(' ')
X_spam_assasin = np.array(spam_assasin['message'])
y_spam_assasin = np.array(spam_assasin['label'])


BIO_ALGS = ['MRFO', 'HGS', 'AO', 'HHO']
ALGS = ["RSCV", "DEFAULT"] + BIO_ALGS
# ALGS = BIO_ALGS


def resolve_dataset(name):
    if (name == 'enron'):
        return [X_enron.copy(), y_enron.copy()]
    elif (name == 'ling_spam'):
        return [X_ling_spam.copy(), y_ling_spam.copy()]
    elif (name == 'spam_assasin'):
        return [X_spam_assasin.copy(), y_spam_assasin.copy()]
    else:
        return


def resolve_alg(alg):
    if alg == 'AO':
        return AO.OriginalAO
    elif alg == 'HGS':
        return HGS.OriginalHGS
    elif alg == 'SSA':
        return SSA.OriginalSSA
    elif alg == 'MRFO':
        return MRFO.BaseMRFO
    elif alg == 'HHO':
        return HHO.BaseHHO


def bio(alg, X, y):
    alg = resolve_alg(alg)
    cv = TfidfVectorizer(stop_words=stopwords.words('english'))
    skf = StratifiedKFold(n_splits=10, random_state=None, shuffle=False)

    alpha, epsilon, tol = [], [], []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train = cv.fit_transform(X_train)
        X_test = cv.transform(X_test)

        def obj_function(solution):
            alpha, epsilon, tol = solution
            clf = SGDClassifier(random_state=0, alpha=alpha,
                                epsilon=epsilon, tol=tol, n_jobs=-1)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            return accuracy_score(y_test, y_pred)

        problem = {
            'obj_func': obj_function,
            'lb': [0.0001, 0.0001, 0.0001],
            'ub': [1000, 1000, 1000],
            'minmax': 'max',
            'verbose': True,
        }

        model = alg(problem, epoch=10, pop_size=40)
        model.solve()
        a, e, t = model.g_best[0]
        alpha.append(a)
        epsilon.append(e)
        tol.append(t)

    return [mean(alpha), mean(epsilon), mean(tol)]


def get_best(alg, X, y):
    if (alg == 'RSCV'):
        distributions = {
            'clf__epsilon': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'clf__alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'clf__tol': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
        }
        clf = Pipeline([
            ('tfidf_vectorizer', TfidfVectorizer(
                stop_words=stopwords.words('english'))),
            ('clf', SGDClassifier(random_state=0, n_jobs=-1))])

        clf_random = RandomizedSearchCV(
            clf, distributions, scoring='accuracy', cv=10, random_state=0)
        clf_random.fit(X, y)
        best = clf_random.best_params_

        return [best['clf__alpha'], best['clf__epsilon'], best['clf__tol']]

    elif alg == 'DEFAULT':
        return [0.0001, 0.1, 1e-3]

    return bio(alg, X, y)


def test(train, test):
    [X, y] = resolve_dataset(train)
    [X2, y2] = resolve_dataset(test)

    print(f'Train: {train}, test: {test}')

    for alg in ALGS:
        alpha, epsilon, tol = get_best(alg, X, y)

        clf = Pipeline([
            ('tfidf_vectorizer', TfidfVectorizer(
                stop_words=stopwords.words('english'))),
            ('clf', SGDClassifier(random_state=0, alpha=alpha, epsilon=epsilon, tol=tol, n_jobs=-1))])

        accuracy = cross_val_score(clf, X, y, cv=10)

        clf.fit(X, y)
        y_score = clf.decision_function(X2)
        y_pred = clf.predict(X2)

        # print('Params', alpha, epsilon, tol)
        # print(f'Alg: {alg}')
        # print(f'Train Accuracy mean: {mean(accuracy)}')
        print(f'Accuracy: {accuracy_score(y2, y_pred)}')
        print(f'Confusion matrix {confusion_matrix(y2, y_pred)}')
        print(f'ROC: {roc_auc_score(y2, y_pred)}')
        print()

        y_fpr, y_tpr, _ = roc_curve(y2, y_score)
        pyplot.plot(y_fpr, y_tpr, marker='.', label=alg)

    ns_probs = [0 for _ in range(len(y))]
    ns_fpr, ns_tpr, _ = roc_curve(y, ns_probs)

    pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='Без навыков')
    pyplot.xlabel('Ошибка первого рода')
    pyplot.ylabel('Чувствительность')
    pyplot.legend()
    pyplot.show()


# with open('log', 'a') as f:
#     with redirect_stdout(f):
test('spam_assasin', 'ling_spam')
# f.close()

# test('spam_assasin')

# print(f'Pred ROC aut score: {roc_auc_score(y2, y2_score)}')

# ns_probs = [0 for _ in range(len(y2))]
# ns_auc = roc_auc_score(y2, ns_probs)
# y2_auc = roc_auc_score(y2, y2_score)
# y_auc = roc_auc_score(y_train, y_score)
# y_test_auc = roc_auc_score(y_test, y_test_score)

# print(f'No Skill: ROC AUC={ns_auc}')
# print(f'Applied: ROC AUC={y2_auc}')
# print(f'Trained: ROC AUC={y_auc}')
# print(f'Trained: ROC AUC={y_test_auc}')

# ns_fpr, ns_tpr, _ = roc_curve(y2, ns_probs)
# y2_fpr, y2_tpr, _ = roc_curve(y2, y2_score)
# y_fpr, y_tpr, _ = roc_curve(y_train, y_score)
# y_test_fpr, y_test_tpr, _ = roc_curve(y_test, y_test_score)

# pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='Без навыков')
# pyplot.plot(y2_fpr, y2_tpr, marker='.', label='Проверка')
# pyplot.plot(y_fpr, y_tpr, marker='.', label='train')
# pyplot.plot(y_test_fpr, y_test_tpr, marker='.', label='test')

# pyplot.xlabel('Ошибка первого рода')
# pyplot.ylabel('Чувствительность')
# pyplot.legend()
# pyplot.show()
