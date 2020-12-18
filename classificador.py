import json
import pandas as pd
import time
import re
from collections import Counter
import os.path
import sklearn.metrics as metrics
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import multilabel_confusion_matrix, classification_report, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset
from skmultilearn.model_selection import iterative_train_test_split, IterativeStratification
from skmultilearn.model_selection.measures import get_combination_wise_output_matrix
from sklearn.neural_network import MLPClassifier
from skmultilearn.adapt import MLTSVM, MLkNN, MLARAM
from sklearn.linear_model import LogisticRegression
from skmultilearn.cluster.networkx import NetworkXLabelGraphClusterer
from skmultilearn.cluster import LabelCooccurrenceGraphBuilder
from skmultilearn.ensemble import MajorityVotingClassifier, LabelSpacePartitioningClassifier, RakelD
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from gensim.sklearn_api import D2VTransformer
from tokenizador import Tokenizador
from sklearn.utils import shuffle
import numpy as np

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)  

def read_json(filename):
    with open(filename) as f:
        return json.load(f)

def dump_results(filename, results):
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)

def run(normas, models, clfs, n_jobs=1):
    # Corpus e labels:
    corpus = [norma['TextoPreProcessado'] for norma in normas]
    labels = [norma['AssuntoGeral'] for norma in normas]

    # Obtém X e y:
    X = corpus
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(labels)

    # Faz shuffle:
    X, y = shuffle(X, y, random_state=42)

    for model in models:
        for clf in clfs:
            fileName = f'{model[-1][0]}_{clf["clf"][0]}.csv'
            if os.path.isfile(fileName):
                print(f'{fileName} já existe! Ignorando!')
                continue
                
            start = time.time()
            pipeline = Pipeline([*model, clf['clf']])

            gs = GridSearchCV(pipeline, clf['params'], scoring='f1_micro', return_train_score=True, cv=IterativeStratification(n_splits=2, order=1), n_jobs=n_jobs, verbose=10)
            gs.fit(X, y)

            print('tempo gasto: ',round(time.time()-start,0),'segundos')
            print('melhores parâmetros:', gs.best_params_, 'melhor score: ', gs.best_score_)

            dump_results(fileName, gs.cv_results_)


def run_test1(normas):
    models = [
        [('cv', CountVectorizer(min_df=20, max_df=0.5))],
        [('tfidf', TfidfVectorizer(min_df=20, max_df=0.5))],
        [('tokenize', Tokenizador()), ('d2v', D2VTransformer(dm=0, min_count=100, size=200, workers=6))]
    ]

    clfs = [
        {
            'clf': ('dt', DecisionTreeClassifier()),
            'params': {
                'dt__min_samples_split': [0.005, 0.010, 2],
                'dt__max_depth': [16, 32, None]
            }
        },
        {
            'clf': ('rf', RandomForestClassifier()),
            'params': {
                'rf__n_estimators': [100, 110, 120],
                'rf__min_samples_split': [0.005, 0.010, 2],
                'rf__min_samples_leaf': [5, 3, 1]
            }
        },
        {
            'clf': ('mlknn', MLkNN()), 
            'params': {
                'mlknn__k': [6, 8, 10, 12], 
                'mlknn__s': [0.5, 1.0, 1.5, 2.0]
            }
        },
        {
            'clf': ('mlp', MLPClassifier()), 
            'params': {
                'mlp__hidden_layer_sizes': [(150), (100, 100), (50, 50, 50)],
                'mlp__activation': ['tanh', 'relu'],
                'mlp__solver': ['sgd', 'adam']
            }
        }
    ]
    run(normas, models, clfs)

def run_test2(normas, n_jobs=1):
    models = [
        [('tfidf', TfidfVectorizer(min_df=20, max_df=0.5))]
    ]

    clfs = [
        {
            'clf': ('mv_mlp', MajorityVotingClassifier(classifier=MLPClassifier())), 
            'params': {
                'mv_mlp__classifier__hidden_layer_sizes': [(150), (100, 100), (50, 50, 50)],
                'mv_mlp__classifier__activation': ['tanh', 'relu'],
                'mv_mlp__clusterer' : [
                    NetworkXLabelGraphClusterer(LabelCooccurrenceGraphBuilder(weighted=True, include_self_edges=False), 'louvain'),
                    NetworkXLabelGraphClusterer(LabelCooccurrenceGraphBuilder(weighted=True, include_self_edges=False), 'lpa')
                ]
            }
        }
    ]
    run(normas, models, clfs, n_jobs)

def run_test3(normas, n_jobs=1):
    # Corpus e labels:
    corpus = [norma['TextoPreProcessado'] for norma in normas]
    labels = [norma['AssuntoGeral'] for norma in normas]

    # Obtém X e y:
    X = corpus
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(labels)
    
    # Faz shuffle:
    X, y = shuffle(X, y, random_state=42)

    # Vectorizer
    X = TfidfVectorizer(min_df=20, max_df=0.5).fit_transform(X).toarray()

    # TrainTestSplit:
    X_train, y_train, X_test, y_test = iterative_train_test_split(X, y, test_size = 0.5)

    # Classifcador:
    clf = MLPClassifier(hidden_layer_sizes=(150), activation='relu')
    clf.fit(X_train, y_train)

    # Prevê
    y_pred = clf.predict(X_test)


def run_test4(normas, n_jobs=1):
    # Corpus e labels:
    corpus = [norma['TextoPreProcessado'] for norma in normas]
    labels = [norma['AssuntoGeral'] for norma in normas]

    # Obtém X e y:
    X = corpus
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(labels)

    # Faz shuffle:
    X, y = shuffle(X, y, random_state=42)

    # Vectorizer
    X = TfidfVectorizer(min_df=20, max_df=0.5).fit_transform(X).toarray()

    def create_model():
        model = Sequential()
        model.add(Input(X.shape[1]))
        model.add(Dense(150, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(y.shape[1], activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
        return model

    # Faz cross-validação:
    results = cross_validate(KerasClassifier(build_fn=create_model, epochs=10, batch_size=200, verbose=1), X=X, y=y, return_train_score=True, cv=IterativeStratification(n_splits=2, order=1), n_jobs=n_jobs, verbose=10)
    print(results)


def main():
    normas = read_json("normas_80assuntos_processadas.json")
    run_test3(normas, n_jobs=2)

if __name__ == "__main__":
    main()