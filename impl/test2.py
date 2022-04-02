import statistics
from mealpy.swarm_based import AO, HGS, SSA, MRFO, HHO
from matplotlib import pyplot
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, auc
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import csv
from scipy.stats import uniform, truncnorm, randint
import numpy as np
import pandas as pd

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


def test_bio_alg(alg, X_train, y_train):
    def obj_function(solution):
        alpha, epsilon, tol = solution
        clf = SGDClassifier(random_state=0, alpha=alpha,
                            epsilon=epsilon, tol=tol, n_jobs=-1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_train)
        return accuracy_score(y_train, y_pred)

    alg = resolve_alg(alg)

    problem = {
        'obj_func': obj_function,
        'lb': [0.0001, 0.0001, 0.0001],
        'ub': [1000, 1000, 1000],
        'minmax': 'max',
        'verbose': True,
    }

    model = alg(problem, epoch=10, pop_size=40)
    model.solve()
    return model.g_best


def get_best(alg, dataset):
    [X, y] = resolve_dataset(dataset)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    cv = TfidfVectorizer(stop_words=stopwords.words('english'))
    X_train = cv.fit_transform(X_train)
    X_test = cv.transform(X_test)

    if (alg == 'RSCV'):
        distributions = {
            # 'epsilon': uniform(0.0001, 1000),
            # 'alpha': uniform(0.0001, 1000),
            # 'tol': uniform(0.0001, 1000),
            'epsilon': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'tol': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
            # 'tfidf__max_df': np.linspace(0.1, 1, 10),
            # 'tfidf__binary': [True, False],
            # 'tfidf__norm': [None, 'l1', 'l2'],
        }

        clf = SGDClassifier(random_state=0, n_jobs=-1)

        model = RandomizedSearchCV(
            clf, distributions, scoring='accuracy', random_state=0)

        model.fit(X_train, y_train)
        params = model.best_params_

        return [params['alpha'], params['epsilon'], params['tol']]

    elif alg == "DEFAULT":
        return [0.0001, 0.1, 1e-3]

    best_params_ = test_bio_alg(alg, X_train, y_train, X_test, y_test)

    return best_params_[0]


def test(train, test, alg):
    [X, y] = resolve_dataset(train)
    [X2, y2] = resolve_dataset(test)

    params = get_best(alg, train)

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25)

    clf = Pipeline([
        ('tfidf_vectorizer', TfidfVectorizer(
            stop_words=stopwords.words('english'))),
        ('classificator', SGDClassifier(random_state=0, alpha=params[0], epsilon=params[1], tol=params[2], n_jobs=-1))])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    y2_pred = clf.predict(X2)

    test_acc = accuracy_score(y2, y2_pred)
    test_prec = precision_score(y2, y2_pred)
    test_recall = recall_score(y2, y2_pred)
    test_f1 = f1_score(y2, y2_pred)

    print(f'Acc: {acc} Prec: {prec} Recall: {recall} F1: {f1}')
    print(
        f'AccTest: {test_acc} PrecTest: {test_prec} RecallTest: {test_recall} F1Test: {test_f1}')


# test(train='spam_assasin', test='ling_spam', alg='DEFAULT')
# test(train='spam_assasin', test='ling_spam', alg='RSCV')
test(train='spam_assasin', test='ling_spam', alg='DEFAULT')
test(train='spam_assasin', test='ling_spam', alg='RSCV')
test(train='spam_assasin', test='ling_spam', alg='MRFO')
