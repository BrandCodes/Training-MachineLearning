# Importar las bibliotecas necesarias
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

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

# Definir el espacio de búsqueda de hiperparámetros
param_grid = {
    'n_estimators': [10, 50, 100],       # Número de árboles
    'max_depth': [None, 10, 20],         # Profundidad máxima
    'min_samples_split': [2, 5, 10],     # Mínimas muestras para dividir
    'min_samples_leaf': [1, 2, 4],       # Mínimas muestras en una hoja
    'max_features': ['sqrt', 'log2'],    # Número máximo de características consideradas
}

# 4. Configurar GridSearchCV
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,  # Validación cruzada con 5 particiones
    scoring='accuracy',  # Métrica de evaluación
    n_jobs=-1,  # Usar todos los núcleos disponibles
    verbose=2  # Imprimir detalles del proceso
)

# 5. Entrenar el modelo con búsqueda de hiperparámetros
print("Iniciando búsqueda con GridSearchCV...")
grid_search.fit(X_train, y_train)

# 6. Mostrar los mejores hiperparámetros
print("\nMejores hiperparámetros encontrados:")
print(grid_search.best_params_)

# 7. Evaluar el mejor modelo en el conjunto de prueba
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nPrecisión en el conjunto de prueba: {accuracy * 100:.2f}%")

# 8. Reporte de clasificación
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=['Besni', 'Kecimen']))
