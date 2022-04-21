from sklearn.model_selection import StratifiedKFold
from mealpy.swarm_based import AO, HGS, SSA, MRFO
from matplotlib import pyplot
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold, RandomizedSearchCV
import numpy as np
import pandas as pd
from numpy import mean

DEFAULT_PARAMS = [0.0001, 0.1, 1e-3]

EN = pd.read_csv(f'./input/enron/messages.csv').fillna(' ')
X_EN = np.array(EN['message'])
y_EN = np.array(EN['label'])

LS = pd.read_csv(f'./input/ling_spam/messages.csv').fillna(' ')
X_LS = np.array(LS['message'])
y_LS = np.array(LS['label'])

SA = pd.read_csv(f'./input/spam_assasin/messages.csv').fillna(' ')
X_SA = np.array(SA['message'])
y_SA = np.array(SA['label'])


def resolve_dataset(name):
    if (name == 'EN'):
        return [X_EN.copy(), y_EN.copy()]
    elif (name == 'LS'):
        return [X_LS.copy(), y_LS.copy()]
    elif (name == 'SA'):
        return [X_SA.copy(), y_SA.copy()]
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


def alg_to_label(alg):
    if alg == 'AO':
        return 'Оптимизатор орла (AO)'
    elif alg == 'HGS':
        return 'Поиск голодных игр (HGS)'
    elif alg == 'SSA':
        return 'Алгоритм поиска воробьев (SSA)'
    elif alg == 'MRFO':
        return 'Алгоритм кормодобывания \nскатов манта (MRFO)'
    elif alg == 'RSCV':
        return 'Случайный поиск'
    elif alg == 'DEFAULT':
        return 'Параметры по умолчанию'


def get_best(alg, X, y):
    if (alg == 'RSCV'):
        distributions = {
            'clf__epsilon': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'clf__alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'clf__tol': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
        }
        skf = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
        clf = Pipeline([
            ('tfidf_vectorizer', TfidfVectorizer(
                stop_words=stopwords.words('english'))),
            ('clf', SGDClassifier(random_state=0, class_weight='balanced', n_jobs=-1))])

        clf_random = RandomizedSearchCV(
            clf, distributions, scoring='accuracy', cv=skf, random_state=0)
        clf_random.fit(X, y)
        best = clf_random.best_params_

        return [best['clf__alpha'], best['clf__epsilon'], best['clf__tol']]

    alg = resolve_alg(alg)
    cv = TfidfVectorizer(stop_words=stopwords.words('english'))
    skf = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)

    alpha, epsilon, tol = [], [], []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train = cv.fit_transform(X_train)
        X_test = cv.transform(X_test)

        def obj_function(solution):
            alpha, epsilon, tol = solution
            clf = SGDClassifier(random_state=0, class_weight='balanced', alpha=alpha,
                                epsilon=epsilon, tol=tol, n_jobs=-1)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            return accuracy_score(y_test, y_pred)

        problem = {
            'fit_func': obj_function,
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


def test(clfs, train, test):
    print(f'Test models trained with {train} on {test}')

    for alg in clfs:
        X, y = resolve_dataset(test)
        clf = clfs[alg]

        y_score = clf.decision_function(X)
        y_pred = clf.predict(X)

        print(alg)
        print('Accuracy %.5f: ' % accuracy_score(y, y_pred))
        print('ROC: %.5f' % roc_auc_score(y, y_score))
        print('F1: %.5f\n' % f1_score(y, y_pred))

        y_fpr, y_tpr, _ = roc_curve(y, y_score)
        pyplot.plot(y_fpr, y_tpr, marker=',', label=alg_to_label(
            alg) + ', ROC: %.5f' % roc_auc_score(y, y_score))

    ns_probs = [0 for _ in range(len(y))]
    ns_fpr, ns_tpr, _ = roc_curve(y, ns_probs)
    pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='Без навыков')
    pyplot.xlabel('Ошибка первого рода')
    pyplot.ylabel('Чувствительность')
    pyplot.legend()
    pyplot.show()


pyplot.rcParams['figure.figsize'] = [30, 15]
pyplot.rcParams.update({'font.size': 30})


best_LS_RSCV = get_best("RSCV", X_LS, y_LS)
best_LS_MRFO = get_best("MRFO", X_LS, y_LS)
best_LS_HGS = get_best("HGS", X_LS, y_LS)
best_LS_AO = get_best("AO", X_LS, y_LS)
best_LS_SSA = get_best("SSA", X_LS, y_LS)

best_SA_RSCV = get_best("RSCV", X_SA, y_SA)
best_SA_MRFO = get_best("MRFO", X_SA, y_SA)
best_SA_HGS = get_best("HGS", X_SA, y_SA)
best_SA_AO = get_best("AO", X_SA, y_SA)
best_SA_SSA = get_best("SSA", X_SA, y_SA)

best_EN_RSCV = get_best("RSCV", X_EN, y_EN)
best_EN_MRFO = get_best("MRFO", X_EN, y_EN)
best_EN_HGS = get_best("HGS", X_EN, y_EN)
best_EN_AO = get_best("AO", X_EN, y_EN)
best_EN_SSA = get_best("SSA", X_EN, y_EN)


LS_EN = {1.: 10, 0.: 1}
LS_SA = {1.: 1, 0.: 1.5}

SA_EN = {1.: 10, 0.: 1}
SA_LS = {1.: 1, 0.: 8}

EN_LS = {1.: 1, 0.: 100}
EN_SA = {1.: 1, 0.: 100}


def create_clf(params, class_weight=None):
    alpha, epsilon, tol = params
    return Pipeline([
        ('tfidf_vectorizer', TfidfVectorizer(
            stop_words=stopwords.words('english'))),
        ('clf', SGDClassifier(random_state=0,
         alpha=alpha, epsilon=epsilon, tol=tol, class_weight=class_weight, n_jobs=-1))
    ])


X, y = X_SA, y_SA

SA_clfs = {
    "RSCV": create_clf(best_SA_RSCV),
    "DEFAULT": create_clf(DEFAULT_PARAMS),
    "MRFO": create_clf(best_SA_MRFO),
    "HGS": create_clf(best_SA_HGS),
    "AO": create_clf(best_SA_AO),
    "SSA": create_clf(best_SA_SSA)
}

for alg in SA_clfs:
    print(alg, mean(cross_val_score(
        SA_clfs[alg], X, y, cv=10, scoring='accuracy')))

SA_EN_clfs = {
    "RSCV": create_clf(best_SA_RSCV, SA_EN),
    "DEFAULT": create_clf(DEFAULT_PARAMS, SA_EN),
    "MRFO": create_clf(best_SA_MRFO, SA_EN),
    "HGS": create_clf(best_SA_HGS, SA_EN),
    "AO": create_clf(best_SA_AO, SA_EN),
    "SSA": create_clf(best_SA_SSA, SA_EN)
}

for alg in SA_EN_clfs:
    SA_EN_clfs[alg].fit(X_SA, y_SA)

test(SA_EN_clfs, "SA", "EN")

SA_LS_clfs = {
    "RSCV": create_clf(best_SA_RSCV, SA_LS),
    "DEFAULT": create_clf(DEFAULT_PARAMS, SA_LS),
    "MRFO": create_clf(best_SA_MRFO, SA_LS),
    "HGS": create_clf(best_SA_HGS, SA_LS),
    "AO": create_clf(best_SA_AO, SA_LS),
    "SSA": create_clf(best_SA_SSA, SA_LS)
}

for alg in SA_LS_clfs:
    SA_LS_clfs[alg].fit(X_SA, y_SA)

test(SA_LS_clfs, "SA", "LS")

X, y = X_LS, y_LS

LS_clfs = {
    "RSCV": create_clf(best_LS_RSCV),
    "DEFAULT": create_clf(DEFAULT_PARAMS),
    "MRFO": create_clf(best_LS_MRFO),
    "HGS": create_clf(best_LS_HGS),
    "AO": create_clf(best_LS_AO),
    "SSA": create_clf(best_LS_SSA)
}

for alg in LS_clfs:
    print(alg, mean(cross_val_score(
        LS_clfs[alg], X, y, cv=10, scoring='accuracy')))

LS_EN_clfs = {
    "RSCV": create_clf(best_LS_RSCV, LS_EN),
    "DEFAULT": create_clf(DEFAULT_PARAMS, LS_EN),
    "MRFO": create_clf(best_LS_MRFO, LS_EN),
    "HGS": create_clf(best_LS_HGS, LS_EN),
    "AO": create_clf(best_LS_AO, LS_EN),
    "SSA": create_clf(best_LS_SSA, LS_EN)
}

for alg in LS_EN_clfs:
    LS_EN_clfs[alg].fit(X_LS, y_LS)

test(LS_EN_clfs, "LS", "EN")

LS_SA_clfs = {
    "HGS": create_clf(best_LS_HGS, LS_SA),
    "DEFAULT": create_clf(DEFAULT_PARAMS, LS_SA),
    "RSCV": create_clf(best_LS_RSCV, LS_SA),
    "MRFO": create_clf(best_LS_MRFO, LS_SA),
    "AO": create_clf(best_LS_AO, LS_SA),
    "SSA": create_clf(best_LS_SSA, LS_SA)
}

for alg in LS_SA_clfs:
    LS_SA_clfs[alg].fit(X_LS, y_LS)

test(LS_SA_clfs, "LS", "SA")

EN_LS_clfs = {
    "HGS": create_clf(best_EN_HGS, EN_LS),
    "DEFAULT": create_clf(DEFAULT_PARAMS, EN_LS),
    "RSCV": create_clf(best_EN_RSCV, EN_LS),
    "MRFO": create_clf(best_EN_MRFO, EN_LS),
    "AO": create_clf(best_EN_AO, EN_LS),
    "SSA": create_clf(best_EN_SSA, EN_LS)
}

for alg in EN_LS_clfs:
    EN_LS_clfs[alg].fit(X_EN, y_EN)

test(EN_LS_clfs, "EN", "LS")


EN_SA_clfs = {
    "HGS": create_clf(best_EN_HGS, EN_SA),
    "DEFAULT": create_clf(DEFAULT_PARAMS, EN_SA),
    "RSCV": create_clf(best_EN_RSCV, EN_SA),
    "MRFO": create_clf(best_EN_MRFO, EN_SA),
    "AO": create_clf(best_EN_AO, EN_SA),
    "SSA": create_clf(best_EN_SSA, EN_SA)
}

for alg in EN_SA_clfs:
    EN_SA_clfs[alg].fit(X_EN, y_EN)

test(EN_SA_clfs, "EN", "SA")

X, y = X_EN, y_EN

EN_clfs = {
    "RSCV": create_clf(best_EN_RSCV),
    "DEFAULT": create_clf(DEFAULT_PARAMS),
    "MRFO": create_clf(best_EN_MRFO),
    "HGS": create_clf(best_EN_HGS),
    "AO": create_clf(best_EN_AO),
    "SSA": create_clf(best_EN_SSA)
}

for alg in EN_clfs:
    print(alg, mean(cross_val_score(
        EN_clfs[alg], X, y, cv=10, scoring='accuracy')))
