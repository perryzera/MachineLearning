from flask import Flask, render_template, request
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import io
import base64
import matplotlib 

matplotlib.use('Agg')

iris = load_iris()
X = iris.data
y = iris.target

from base import app

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/classify', methods=['POST'])
def classify():

    classifier_type = request.form['classifier']

    if classifier_type == 'decision_tree':
        max_depth = int(request.form.get('max_depth', 4))   
        min_samples_split = max(int(request.form.get('min_samples_split', 2)), 2)
        min_samples_leaf = int(request.form.get('min_samples_leaf', 1))
        clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=0)

    elif classifier_type == 'knn':
        n_neighbors = int(request.form.get('n_neighbors', 5))
        weights = request.form.get('weights', 'uniform')
        algorithm = request.form.get('algorithm', 'auto')
        clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)

    elif classifier_type == 'svm':
        C = float(request.form.get('C', 1.0))
        kernel = request.form.get('kernel', 'rbf')
        gamma = request.form.get('gamma', 'scale')
        clf = SVC(C=C, kernel=kernel, gamma=gamma)

    elif classifier_type == 'mlp':
        hidden_layer_sizes = tuple(map(int, request.form.get('hidden_layer_sizes', '10,10').split(',')))
        activation = request.form.get('activation', 'relu')
        alpha = float(request.form.get('alpha', 0.0001))
        clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, alpha=alpha, random_state=0)

    else:
        return "Classificador não reconhecido."

    x_treino, x_teste, y_treino, y_teste = train_test_split(X, y, random_state=0)
    clf.fit(x_treino, y_treino)

    y_pred = clf.predict(x_teste)

    accuracy = accuracy_score(y_teste, y_pred)
    classification_rep = classification_report(y_teste, y_pred)

    cv_results = cross_val_score(clf, X, y, cv=5, scoring='accuracy')

    plt.switch_backend('Agg')

    plt.bar(range(1, 6), cv_results)
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.title('Cross-Validation Results')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return render_template('result.html', classifier=classifier_type, accuracy=accuracy, classification_report=classification_rep, plot_url=plot_url)

@app.route('/result')
def result():
    return render_template('result.html')
