# Proyecto de Machine Learning   
### Predicción de Precios de Casas  
### Clasificación de pasas (y diferentes técnicas de optimización) 

Estos proyectos utilizan técnicas básicas de Machine Learning para predecir segun los datos dentro de cada Dataset

## Requisitos Previos

Antes de comenzar, asegúrate de tener instalado:

- Python 3.8 o superior
- `pip` para instalar las bibliotecas requeridas
- Un IDE como Visual Studio Code (opcional, pero recomendado)

## Instalación

Sigue estos pasos para configurar el proyecto en tu máquina:

1. Clona este repositorio:
   <!-- ```bash
   git clone https://github.com/
   cd proyecto-ml -->

## Uso

Este modelo sirve como ejemplo para el aprendizaje de manejo y uso de Data Sets y conocer las predicciones resultantes segun los datos proporcionados.

## Ejemplos disponibles:
-Predicción de precios de casas [Ejemplo básico: Predicción de precios de casas]  
Descripción:  
Usaremos un dataset simple de precios de casas para entrenar un modelo de regresión lineal que prediga el precio basado en características como el tamaño.  
Explicación:  
   -Crea datos simples de casas (tamaño y precio).  
   -Visualiza los datos para identificar patrones.  
   -Entrena un modelo de regresión lineal usando scikit-learn.  
   -Evalúa el modelo con Mean Squared Error (MSE).  
   -Dibuja la línea de regresión sobre los datos reales.  
   -Realiza una predicción sobre un tamaño nuevo.  

-Clasificación de pasas [Ejemplo básico: Raisin Dataset (Kaggle)]  
El objetivo es clasificar las pasas en una de las dos clases.  
DS:  
   -Area: Área de la pasa.  
   -Perimeter: Perímetro.  
   -MajorAxisLength: Longitud del eje mayor.  
   -MinorAxisLength: Longitud del eje menor.  
   -Eccentricity: Excentricidad.  
   -Class: Tipo de pasa (Kecimen o Besni).  
 Explicación:  
   -Cargar el dataset: Leemos el archivo CSV del Raisin Dataset.  
   -Exploración inicial: Revisamos las primeras filas y la estructura del dataset (info()).  
   -Preprocesamiento:  
      Convertimos la columna Class (categórica) a valores numéricos (0 y 1).  
      Separamos las características (X) de la etiqueta (y).  
   -Entrenamiento del modelo:  
      Usamos un modelo de clasificación Random Forest.  
      Dividimos los datos en 80% para entrenamiento y 20% para prueba.  
   -Evaluación:  
      Calculamos la precisión (accuracy_score) del modelo.  
      Mostramos un reporte de clasificación y una matriz de confusión para evaluar el rendimiento.  
   -Predicción de ejemplo: Realizamos una predicción con un registro del conjunto de prueba.  

