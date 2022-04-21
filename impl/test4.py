from sklearn.naive_bayes import MultinomialNB
from audioop import cross
from contextlib import redirect_stdout
from sklearn.model_selection import StratifiedKFold
from mealpy.swarm_based import AO, HGS, SSA, MRFO, HHO
from matplotlib import pyplot
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, auc
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV
import numpy as np
import pandas as pd
from numpy import mean
from numpy import std

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


# pyplot.rcParams['figure.figsize'] = [80, 40]
# pyplot.rcParams.update({'font.size': 60})

# test clfs trained with <train> dataset on <test> dataset


def test(clfs, train, test):
    print(f'Test models trained with {train} on {test}')

    for alg in clfs:
        X, y = resolve_dataset(test)
        clf = clfs[alg]

        y_score = clf.decision_function(X)
        y_pred = clf.predict(X)

        print(alg)
        print('Accuracy %.5f: ' % accuracy_score(y, y_pred))
        print('ROC: %.5f' % roc_auc_score(y, y_pred))
        print('F1: %.5f\n' % f1_score(y, y_pred))

        y_fpr, y_tpr, _ = roc_curve(y, y_score)

        # if alg == 'AO':
        #     pyplot.plot(y_fpr, y_tpr, marker='x', label=alg)
        # elif alg == 'SSA':
        #     pyplot.plot(y_fpr, y_tpr, marker='+', label=alg)
        # else:
        pyplot.plot(y_fpr, y_tpr, marker=',', label=alg)

    ns_probs = [0 for _ in range(len(y))]
    ns_fpr, ns_tpr, _ = roc_curve(y, ns_probs)

    pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='Без навыков')
    pyplot.xlabel('Ошибка первого рода')
    pyplot.ylabel('Чувствительность')
    pyplot.legend()
    pyplot.show()


def create_clf(params, class_weight=None):
    alpha, epsilon, tol = params
    return Pipeline([
        ('tfidf_vectorizer', TfidfVectorizer(
            stop_words=stopwords.words('english'))),
        ('clf', SGDClassifier(random_state=0,
         alpha=alpha, epsilon=epsilon, tol=tol, class_weight=class_weight, n_jobs=-1))
    ])


X, y = X_LS, y_LS

best_LS_RSCV = [0.001, 1, 1]
best_LS_MRFO = [0.00011623717756157859, 484.27076763414436, 216.5086989934]
best_LS_HGS = [0.0022917872068965736, 19.823049434350512, 90.57441814602444]
best_LS_AO = [0.19002423039394772, 41.07346210476601, 57.552068978376624]
best_LS_SSA = [0.032970495018666765, 183.32506925145373, 117.76141405995111]

# LS_clfs = {
#     "RSCV": create_clf(best_LS_RSCV),
#     "DEFAULT": create_clf(DEFAULT_PARAMS),
#     "MRFO": create_clf(best_LS_MRFO),
#     "HGS": create_clf(best_LS_HGS),
#     "AO": create_clf(best_LS_AO),
#     "SSA": create_clf(best_LS_SSA)
# }

# for alg in LS_clfs:
#     LS_clfs[alg].fit(X_LS, y_LS)

# test(LS_clfs, "LS", "SA")
# test(LS_clfs, "LS", "EN")

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


# X, y = X_SA, y_SA

# best_SA_RSCV = [0.0001, 1, 10]
# best_SA_MRFO = [0.00010870235001272333, 451.9050393554245, 181.09353850833924]
# best_SA_HGS = [0.00011032170757958722,
#                0.00011472906393677556, 2.7412975859147672]
# best_SA_AO = [0.0001, 0.1882828225532926, 0.06256962757736609]
# best_SA_SSA = [0.00010541550010794703, 171.5336819945209, 170.657590549754]

# SA_clfs = {
#     "RSCV": create_clf(best_SA_RSCV),
#     "DEFAULT": create_clf(DEFAULT_PARAMS),
#     "MRFO": create_clf(best_SA_MRFO),
#     "HGS": create_clf(best_SA_HGS),
#     "AO": create_clf(best_SA_AO),
#     "SSA": create_clf(best_SA_SSA)
# }


# for alg in SA_clfs:
#     print(alg, mean(cross_val_score(
#         SA_clfs[alg], X, y, cv=10, scoring='accuracy')))

# for alg in SA_clfs:
#     SA_clfs[alg].fit(X_SA, y_SA)

# test(SA_clfs, "SA", "LS")


# --------------------------------------------
#
# best_EN_RSCV = [0.0001, 1, 10]
# best_EN_MRFO = [0.00010110580912339215, 339.20447492054046, 377.83845464774805]
# best_EN_HGS = [0.00010085619945834547, 154.9162953216401, 116.10455211284992]
# best_EN_AO = [0.0001, 96.33289509195406, 111.70127122768649]
# best_EN_SSA = [0.0001, 109.01548465909111, 97.49608032123167]

# X, y = X_EN, y_EN

# EN_clfs = {
#     "RSCV": create_clf(best_EN_RSCV),
#     "MRFO": create_clf(best_EN_MRFO),
#     "HGS": create_clf(best_EN_HGS),
#     "AO": create_clf(best_EN_AO),
#     "SSA": create_clf(best_EN_SSA),
#     "DEFAULT": create_clf(DEFAULT_PARAMS),
# }

# for alg in EN_clfs:
#     print(alg, mean(cross_val_score(EN_clfs[alg], X, y, scoring='accuracy')))

# for alg in EN_clfs:
#     EN_clfs[alg].fit(X_EN, y_EN)

# test(EN_clfs, "EN", "SA")
