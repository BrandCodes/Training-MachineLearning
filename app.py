# import numpy as np
# print("hao !")
# print("¡Tu entorno de Python está funcionando!")

# Importar bibliotecas necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Crear datos simples de ejemplo
# Supongamos que tenemos un dataset con 'Tamaño (m²)' y 'Precio (USD)'
data = {
    'Tamaño (m²)': [50, 60, 70, 80, 100, 120, 150],
    'Precio (USD)': [150000, 180000, 200000, 230000, 300000, 360000, 450000]
}
df = pd.DataFrame(data)

# 2. Visualizar los datos
plt.scatter(df['Tamaño (m²)'], df['Precio (USD)'], color='blue')
plt.title("Tamaño vs Precio")
plt.xlabel("Tamaño (m²)")
plt.ylabel("Precio (USD)")
plt.show()

# 3. Preparar los datos
X = df[['Tamaño (m²)']]  # Característica (input)
y = df['Precio (USD)']   # Etiqueta (output)

# Dividir los datos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Crear y entrenar el modelo
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Hacer predicciones
y_pred = model.predict(X_test)

# 6. Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
print(f"Error Cuadrático Medio (MSE): {mse:.2f}")

# 7. Visualizar la línea de regresión
plt.scatter(X, y, color='blue', label="Datos reales")
plt.plot(X, model.predict(X), color='red', label="Línea de regresión")
plt.title("Regresión Lineal: Tamaño vs Precio")
plt.xlabel("Tamaño (m²)")
plt.ylabel("Precio (USD)")
plt.legend()
plt.show()

# 8. Predicción de ejemplo
nuevo_tamaño = [[85]]  # Tamaño de una nueva casa
prediccion = model.predict(nuevo_tamaño)
print(f"El precio estimado para una casa de {nuevo_tamaño[0][0]} m² es ${prediccion[0]:,.2f} USD")


# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause