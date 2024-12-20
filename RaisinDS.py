# Importar bibliotecas necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Cargar el dataset
# Reemplaza 'data/raisin.csv' con la ruta de tu archivo [Recibe Excel]
# data = pd.read_csv("/data/raisin.csv")
data = pd.read_excel("data/raisin.xlsx", engine="openpyxl")

# Guardar como archivo .csv
data.to_csv("data/raisin.csv", index=False)

# 2. Explorar los datos
print("Primeras filas del dataset:")
print(data.head())
print("\nResumen de las columnas:")
print(data.info())

# 3. Preprocesamiento
# Convertir la columna 'Class' en numérica (0 = Besni, 1 = Kecimen)
data['Class'] = data['Class'].map({'Besni': 0, 'Kecimen': 1})

# Separar las características (X) y la etiqueta (y)
X = data.drop(columns=['Class'])
y = data['Class']

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Entrenar el modelo
# Usamos un modelo de Random Forest para clasificación
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# 5. Evaluar el modelo
# Predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"\nPrecisión del modelo: {accuracy * 100:.2f}%")

# Mostrar reporte de clasificación
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=['Besni', 'Kecimen']))

# Matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualizar la matriz de confusión
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Besni', 'Kecimen'], yticklabels=['Besni', 'Kecimen'])
plt.title("Matriz de Confusión")
plt.xlabel("Predicción")
plt.ylabel("Verdadero")
plt.show()

# 6. Predicción de ejemplo
# Tomar un ejemplo del conjunto de prueba
ejemplo = X_test.iloc[0]
prediccion = model.predict([ejemplo])
clase_predicha = 'Kecimen' if prediccion[0] == 1 else 'Besni'
print(f"\nPredicción para el ejemplo: {clase_predicha}")
