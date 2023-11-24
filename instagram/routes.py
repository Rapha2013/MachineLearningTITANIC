
from flask import Flask, render_template, request, redirect, url_for, jsonify
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from instagram import app

url = './data/titanic.csv'
titanic = pd.read_csv(url)
# titanic.head()

columns_to_drop = ['Name', 'PassengerId', 'Cabin', 'Embarked',
                   'SibSp', 'Parch', 'Ticket', 'Fare']

for column in columns_to_drop:
  titanic = titanic.drop(column, axis=1)

for column in ['Age', 'Sex', 'Pclass']:
  titanic = titanic[titanic[column].notna()]

sex_int = {'male': 0, 'female': 1}
titanic['Sex'] = titanic['Sex'].map(sex_int)


X = titanic.drop('Survived', axis=1)
y = titanic['Survived']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Função para gerar a matriz de confusão como uma imagem base64
def plot_confusion_matrix(cm):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    plt.colorbar(im)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# Página inicial
@app.route('/')
def index():
    return render_template('home.html')


# Página inicial
@app.route('/machine', methods=['POST'])
def machine():
     # Recebe dados do JSON
    data = request.json

    # Verifica o tipo de classificador e ajusta os parâmetros correspondentes
    if data['type'] == 'knn':
        knn_classifier = KNeighborsClassifier(n_neighbors=int(data['parameters']['n_neighbors']), weights=data['parameters']['weights'], p=int(data['parameters']['p']))
        knn_classifier.fit(X_train, y_train)
        y_pred_knn = knn_classifier.predict(X_test)
        cm_knn = confusion_matrix(y_test, y_pred_knn).tolist()
        accuracy_knn = accuracy_score(y_test, y_pred_knn)
        f1_score_knn = f1_score(y_test, y_pred_knn, average='macro')

        return jsonify({
            'matriz': plot_confusion_matrix(cm_knn),
            'accuracy': accuracy_knn,
            'f1_score': f1_score_knn
        })

    elif data['type'] == 'svm':
                 # Converte o parâmetro 'C' para um número de ponto flutuante
        C = float(data['parameters'].get('C', 1.0))

        svm_classifier = SVC(C=C, kernel=data['parameters']['kernel'])
        svm_classifier.fit(X_train, y_train)
        y_pred_svm = svm_classifier.predict(X_test)
        cm_svm = confusion_matrix(y_test, y_pred_svm).tolist()
        accuracy_svm = accuracy_score(y_test, y_pred_svm)
        f1_score_svm = f1_score(y_test, y_pred_svm, average='macro')

        return jsonify({
            'matriz': plot_confusion_matrix(cm_svm),
            'accuracy': accuracy_svm,
            'f1_score': f1_score_svm
        })

    elif data['type'] == 'mlp':
        mlp_classifier = MLPClassifier(hidden_layer_sizes=(int(data['parameters']['hidden_layer_sizes'])), max_iter=int(data['parameters']['max_iter']))
        mlp_classifier.fit(X_train, y_train)
        y_pred_mlp = mlp_classifier.predict(X_test)
        cm_mlp = confusion_matrix(y_test, y_pred_mlp).tolist()
        accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
        f1_score_mlp = f1_score(y_test, y_pred_mlp, average='macro')

        return jsonify({
            'matriz': plot_confusion_matrix(cm_mlp),
            'accuracy': accuracy_mlp,
            'f1_score': f1_score_mlp
        })

    elif data['type'] == 'dt':
        dt_classifier = DecisionTreeClassifier(max_depth=int(data['parameters']['max_depth']))
        dt_classifier.fit(X_train, y_train)
        y_pred_dt = dt_classifier.predict(X_test)
        cm_dt = confusion_matrix(y_test, y_pred_dt).tolist()
        accuracy_dt = accuracy_score(y_test, y_pred_dt)
        f1_score_dt = f1_score(y_test, y_pred_dt, average='macro')

        return jsonify({
            'matriz': plot_confusion_matrix(cm_dt),
            'accuracy': accuracy_dt,
            'f1_score': f1_score_dt
        })

    elif data['type'] == 'rf':
        rf_classifier = RandomForestClassifier(n_estimators=int(data['parameters']['n_estimators']), max_depth=int(data['parameters']['max_depth']))
        rf_classifier.fit(X_train, y_train)
        y_pred_rf = rf_classifier.predict(X_test)
        cm_rf = confusion_matrix(y_test, y_pred_rf).tolist()
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        f1_score_rf = f1_score(y_test, y_pred_rf, average='macro') 

        return jsonify({
            'matriz': plot_confusion_matrix(cm_rf),
            'accuracy': accuracy_rf,
            'f1_score': f1_score_rf
        })
   

       