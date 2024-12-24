# Importar las bibliotecas necesarias
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# 1. Cargar el dataset
data = pd.read_excel("data/raisin.xlsx", engine="openpyxl")
data.to_csv("data/raisin.csv", index=False)

data['Class'] = data['Class'].map({'Besni': 0, 'Kecimen': 1})  # Convertir a valores numéricos

# 2. Preparar los datos
X = data.drop(columns=['Class'])
y = data['Class']

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Definir el modelo base y los hiperparámetros a buscar
model = RandomForestClassifier(random_state=42)

# Definir los rangos de búsqueda de hiperparámetros
param_distributions = {
    'n_estimators': [int(x) for x in np.linspace(10, 200, 10)],  # Árboles entre 10 y 200
    'max_depth': [None, 10, 20, 30, 40],                        # Profundidad máxima
    'min_samples_split': [2, 5, 10],                           # Muestras mínimas para dividir
    'min_samples_leaf': [1, 2, 4],                             # Muestras mínimas en hoja
    'max_features': ['sqrt', 'log2', None]                     # Número máximo de características
}

# 4. Configurar RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=50,  # Número de combinaciones a probar
    scoring='accuracy',  # Métrica de evaluación
    cv=5,  # Validación cruzada con 5 particiones
    random_state=42,
    n_jobs=-1,  # Usar todos los núcleos disponibles
    verbose=2  # Imprimir detalles del proceso
)

# 5. Entrenar el modelo con RandomizedSearchCV
print("Iniciando búsqueda con RandomizedSearchCV...")
random_search.fit(X_train, y_train)

# 6. Mostrar los mejores hiperparámetros
print("\nMejores hiperparámetros encontrados:")
print(random_search.best_params_)

# 7. Evaluar el mejor modelo en el conjunto de prueba
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nPrecisión en el conjunto de prueba: {accuracy * 100:.2f}%")

# 8. Reporte de clasificación
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=['Besni', 'Kecimen']))
