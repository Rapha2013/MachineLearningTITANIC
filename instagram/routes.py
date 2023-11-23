
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
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
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
    
    data = request.json
   
     # Initialize classifiers
    knn_classifier = KNeighborsClassifier()
    svm_classifier = SVC()
    mlp_classifier = MLPClassifier()
    dt_classifier = DecisionTreeClassifier()
    rf_classifier = RandomForestClassifier()

    # Train the models
    knn_classifier.fit(X_train, y_train)
    svm_classifier.fit(X_train, y_train)
    mlp_classifier.fit(X_train, y_train)
    dt_classifier.fit(X_train, y_train)
    rf_classifier.fit(X_train, y_train)

    # Make predictions
    y_pred_knn = knn_classifier.predict(X_test)
    y_pred_svm = svm_classifier.predict(X_test)
    y_pred_mlp = mlp_classifier.predict(X_test)
    y_pred_dt = dt_classifier.predict(X_test)
    y_pred_rf = rf_classifier.predict(X_test)

      # Matriz de Confusão
    cm_knn = confusion_matrix(y_test, y_pred_knn).tolist()
    cm_svm = confusion_matrix(y_test, y_pred_svm).tolist()
    cm_mlp = confusion_matrix(y_test, y_pred_mlp).tolist()
    cm_dt = confusion_matrix(y_test, y_pred_dt).tolist()
    cm_rf = confusion_matrix(y_test, y_pred_rf).tolist()

   # Calcular Acurácia e F1-Score
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    f1_score_knn = f1_score(y_test, y_pred_knn, average='macro')

    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    f1_score_svm = f1_score(y_test, y_pred_svm, average='macro')

    accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
    f1_score_mlp = f1_score(y_test, y_pred_mlp, average='macro')

    accuracy_dt = accuracy_score(y_test, y_pred_dt)
    f1_score_dt = f1_score(y_test, y_pred_dt, average='macro')

    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    f1_score_rf = f1_score(y_test, y_pred_rf, average='macro')

    if data['type'] == 'knn':
        return jsonify({
            'matriz': plot_confusion_matrix(cm_knn),
            'accuracy': accuracy_knn,
            'f1_score': f1_score_knn
        })

    if data['type'] == 'svm':
        return jsonify({
            'matriz': plot_confusion_matrix(cm_svm),
            'accuracy': accuracy_svm,
            'f1_score': f1_score_svm
        })

    if data['type'] == 'mlp':
        return jsonify({
            'matriz': plot_confusion_matrix(cm_mlp),
            'accuracy': accuracy_mlp,
            'f1_score': f1_score_mlp
        })

    if data['type'] == 'dt':
        return jsonify({
            'matriz': plot_confusion_matrix(cm_dt),
            'accuracy': accuracy_dt,
            'f1_score': f1_score_dt
        })

    if data['type'] == 'rf':
        return jsonify({
            'matriz': plot_confusion_matrix(cm_rf),
            'accuracy': accuracy_rf,
            'f1_score': f1_score_rf
        })
