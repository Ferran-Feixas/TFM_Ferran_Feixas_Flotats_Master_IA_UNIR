# TFM_Ferran_Feixas_Flotats_Master_IA_UNIR

# Predicción del Riesgo de Infarto Cardíaco

Este repositorio contiene scripts en Python para la predicción del riesgo de infarto cardíaco utilizando diversos modelos de aprendizaje automático y redes neuronales profundas. El proyecto está orientado a la aplicación de técnicas de inteligencia artificial en el ámbito de la salud.

## Características
- Preprocesamiento de datos clínicos, incluyendo imputación de valores faltantes, normalización de variables numéricas y codificación de variables categóricas.
- Implementación y comparación de los siguientes modelos de aprendizaje automático:
  - Regresión Logística
  - Random Forest
  - Máquinas de Soporte Vectorial (SVM)
  - Redes Neuronales Profundas (DNN)
- Evaluación de modelos mediante métricas como Precisión, Recall, F1-score y AUC-ROC.
- Análisis de importancia de variables para interpretabilidad mediante SHAP (Shapley Additive Explanations).
- Visualización de resultados a través de gráficos y representaciones gráficas.

## Conjuntos de Datos
Los scripts están diseñados para trabajar con diferentes conjuntos de datos clínicos relacionados con enfermedades cardiovasculares. Algunos ejemplos incluyen:
- `heart_failure_clinical_records_dataset.csv`
- `Heart_disease_cleveland_new.csv`
- `Framingham.csv`
- `Cardio_train.csv`

Los archivos deben ubicarse en el mismo directorio que los scripts o ajustarse según sea necesario.

## Dependencias
El proyecto requiere Python y las siguientes bibliotecas:
- `numpy` - Para cálculos numéricos.
- `pandas` - Para manipulación y análisis de datos.
- `matplotlib` - Para visualización de datos.
- `scikit-learn` - Para modelos de aprendizaje automático y preprocesamiento.
- `imbalanced-learn` - Para manejar conjuntos de datos desbalanceados.
- `tensorflow` - Para la construcción y entrenamiento de redes neuronales profundas.
- `shap` - Para análisis de interpretabilidad y explicabilidad de modelos.

Las dependencias están listadas en el archivo `requirements.txt`.
