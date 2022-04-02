from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from numpy import std
from numpy import mean
from mealpy.swarm_based import AO, HGS, SSA, MRFO, HHO
from matplotlib import pyplot
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, auc
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder

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


# def resolve_clf(alg):
#     if alg == 'AO':
#         return AO.OriginalAO
#     elif alg == 'HGS':
#         return HGS.OriginalHGS
#     elif alg == 'SSA':
#         return SSA.OriginalSSA
#     elif alg == 'MRFO':
#         return MRFO.BaseMRFO
#     elif alg == 'HHO':
#         return HHO.BaseHHO


# KERNEL_ENCODER = LabelEncoder()
# KERNEL_ENCODER.fit(['linear', 'poly', 'rbf', 'sigmoid'])


X_train, X_test, y_train, y_test = train_test_split(
    X_enron, y_enron, test_size=0.3)


# def obj_function(solution):
#     kernel, c, tol = solution
#     kernel_decoded = KERNEL_ENCODER.inverse_transform([int(kernel)])[0]
#     clf_pipeline = Pipeline([
#         ('tfidf_vectorizer', TfidfVectorizer(
#             stop_words=stopwords.words('english'))),
#         ('classificator', SVC(C=c, random_state=1, kernel='kernel_decoded', tol=tol))])

#     clf_pipeline.fit(X_train, y_train)
#     y_pred = clf_pipeline.predict(X_test)

#     return accuracy_score(y_test, y_pred)


# def test_bio_alg(clf, obj_function):
#     problem = {
#         "obj_func": obj_function,
#         "lb": [0, 0.1, 1e-3],
#         "ub": [3.99, 1000, 1000],
#         "minmax": "max",
#     }
#     model = clf(problem, epoch=10, pop_size=40)
#     model.solve()
#     return model.g_best

clf_pipeline = Pipeline([
    ('tfidf_vectorizer', TfidfVectorizer(
        stop_words=stopwords.words('english'))),
    ('classificator', SGDClassifier())])

clf_pipeline.fit(X_train, y_train)
predictions = clf_pipeline.predict(X_train)
score = accuracy_score(y_train, predictions)
print(f"Acurácia de treinamento: {score*100:.2f}%")

predictions_test = clf_pipeline.predict(X_test)
score = accuracy_score(y_test, predictions_test)
print(f"Acurácia de teste: {score*100:.2f}%")
print()

y2_pred = clf_pipeline.predict(X_ling_spam)
# print(clf_pipeline.predict_proba([X_ling_spam[0]]))
print(f"Acurácia: {accuracy_score(y_ling_spam, y2_pred)*100:.2f}%")
