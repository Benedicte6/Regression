from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import numpy as np

# Chargement du dataset
data = pd.read_csv('geo_facteurs.csv')


# Récupération des variables indépendantes et dépendantes
X = data[['Altitude']]
y = data[['Temperature']]

# Subdivision du dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=0)


X_train = np.array(X_train).reshape(-1, 1)
y_train = np.array(y_train).reshape(-1, 1)

# Choix de l'algorithme (Régression Linéaire)
model = LinearRegression()

# Entraînement du modèle avec les données d'entraînement
model.fit(X_train, y_train)

# Test du modèle avec les données de test
accuracy = model.score(X_test, y_test)

# Affichage de la régression linéaire
plt.scatter(X_train, y_train, color='blue', label='Données réelles')
plt.plot(X_test, y_test, color='red', label='Régression linéaire')
plt.xlabel('Altitude')
plt.ylabel('Température')
plt.legend()
plt.show()

# Enregistrement du modèle
joblib.dump(model, 'temperature_regression_model.pkl')


app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    altitude = float(request.form['altitude'])

    # Chargement du modèle enregistré
    model = joblib.load('temperature_regression_model.pkl')

    # Prédiction avec le modèle chargé
    prediction = model.predict([[altitude]])[0]
    
    return render_template('prediction.html', prediction=prediction)




if __name__ == '__main__':
    app.run(debug=True)

# from flask import Flask, render_template, request
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# import joblib
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import OneHotEncoder

# imputer = SimpleImputer(strategy='mean')

# # Chargement du dataset
# data = pd.read_csv('Value_of_Production_E_All_Data_(Normalized).csv', encoding='latin1')

# # Sélection des colonnes pertinentes
# X = data[['Area Code', 'Area Code (M49)']]
# y = data[['Area']]

# # Filtrer les colonnes non numériques
# X_numeric = X.select_dtypes(include='number')

# # Subdivision du dataset
# X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.2, random_state=0)

# # Imputation des données manquantes
# imputer.fit(X_train)
# X_train_imputed = imputer.transform(X_train)
# X_test_imputed = imputer.transform(X_test)

# # Encodage des variables catégorielles
# encoder = OneHotEncoder()
# y_train_encoded = encoder.fit_transform(y_train).toarray()
# y_test_encoded = encoder.transform(y_test).toarray()

# # Choix de l'algorithme (Régression Linéaire)
# model = LinearRegression()

# # Entraînement du modèle avec les données d'entraînement
# model.fit(X_train_imputed, y_train_encoded)

# # Test du modèle avec les données de test
# accuracy = model.score(X_test_imputed, y_test_encoded)

# # Enregistrement du modèle
# joblib.dump(model, 'agricultural_regression_model.pkl')

# app = Flask(__name__)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     temperature = float(request.form['temperature'])
#     humidity = float(request.form['humidity'])

#     # Chargement du modèle enregistré
#     model = joblib.load('agricultural_regression_model.pkl')

#     # Prédiction avec le modèle chargé
#     prediction = model.predict([[temperature, humidity]])[0]
    
#     return render_template('prediction.html', prediction=prediction)

# if __name__ == '__main__':
#     app.run(debug=True)