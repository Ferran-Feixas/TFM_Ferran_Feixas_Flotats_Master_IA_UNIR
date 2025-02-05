# Importar bibliotecas necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import shap


# Leer el archivo CSV
file_path = 'Framingham.csv'
data = pd.read_csv(file_path, delimiter=';')

# Mostrar las primeras filas del archivo
print("Primeras filas del conjunto de datos:")
print(data.head())

# Información general del conjunto de datos
print("\nInformación del conjunto de datos:")
print(data.info())

# Resumen estadístico del conjunto de datos
print("\nResumen estadístico del conjunto de datos:")
print(data.describe())

# Comprobar valores faltantes
print("\nValores faltantes en el conjunto de datos:")
print(data.isnull().sum())

data.dropna(inplace=True)

# Verificar que no haya valores NaN
print("\nValores faltantes después de eliminar filas con NaN:")
print(data.isnull().sum())

# Dividir en variables predictoras y objetivo
X = data.drop(columns=['TenYearCHD'])
y = data['TenYearCHD']

# Identificar variables numéricas y categóricas
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Crear pipelines de preprocesamiento
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combinar los transformadores en un preprocesador
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Modelos a entrenar con manejo de desbalance
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    'SVM': SVC(probability=True, random_state=42, class_weight='balanced')
}

# Entrenamiento y evaluación de modelos
for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('model', model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline.named_steps['model'], 'predict_proba') else None
    
    print(f"Resultados para {name}:")
    print(classification_report(y_test, y_pred, zero_division=0))
    if y_proba is not None:
        print(f"AUC-ROC: {roc_auc_score(y_test, y_proba)}\n")

# Modelo de Redes Neuronales Profundas
# Preprocesar los datos para TensorFlow
preprocessor_nn = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

X_train_nn = preprocessor_nn.fit_transform(X_train)
X_test_nn = preprocessor_nn.transform(X_test)

# Definir el modelo
model_nn = Sequential([
    Dense(64, activation='relu', input_dim=X_train_nn.shape[1]),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compilar el modelo
model_nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Normalizar solo las variables numéricas para el modelo de red neuronal
scaler_nn = StandardScaler()
X_train_nn = scaler_nn.fit_transform(X_train[numerical_features])
X_test_nn = scaler_nn.transform(X_test[numerical_features])

# Entrenamiento de la red neuronal
model_nn.fit(X_train_nn, y_train, epochs=50, batch_size=16, validation_split=0.2, verbose=1)

# Evaluar el modelo
loss, accuracy = model_nn.evaluate(X_test_nn, y_test, verbose=0)
print(f"Resultados de la Red Neuronal Profunda:")
print(f"Accuracy: {accuracy:.2f}")

# Obtener las predicciones como probabilidades
y_proba_nn = model_nn.predict(X_test_nn)

# Convertir las probabilidades en clases binarias con umbral 0.5
y_pred_nn = (y_proba_nn >= 0.5).astype(int)

# Calcular métricas de desempeño
report_nn = classification_report(y_test, y_pred_nn, output_dict=True)

# Extraer precisión, recall y f1-score para cada clase
precision_0_nn = report_nn["0"]["precision"]
recall_0_nn = report_nn["0"]["recall"]
f1_0_nn = report_nn["0"]["f1-score"]

precision_1_nn = report_nn["1"]["precision"]
recall_1_nn = report_nn["1"]["recall"]
f1_1_nn = report_nn["1"]["f1-score"]

# Calcular AUC-ROC
auc_roc_nn = roc_auc_score(y_test, y_proba_nn)

# Mostrar resultados de la Red Neuronal
print(f"Resultados para la Red Neuronal Profunda:")
print(classification_report(y_test, y_pred_nn))
print(f"AUC-ROC: {auc_roc_nn:.2f}")

# Entrenar el modelo de Regresión Logística con el pipeline
pipeline_lr = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', LogisticRegression(max_iter=1000, random_state=42))])
pipeline_lr.fit(X_train, y_train)

# Ajustar el preprocesador directamente en los datos de entrenamiento
preprocessor.fit(X_train)

# Variables numéricas
numerical_features = preprocessor.transformers_[0][2]  # Índices de las variables numéricas

# Variables categóricas codificadas
onehot_encoder = preprocessor.transformers_[1][1].named_steps['onehot']
onehot_encoder.fit(X_train[categorical_features])  # Ajustar el OneHotEncoder a los datos categóricos
encoded_categorical_features = onehot_encoder.get_feature_names_out(categorical_features)

# Combinar nombres de todas las características
feature_names = np.array(list(numerical_features) + list(encoded_categorical_features))

# Obtener los coeficientes del modelo entrenado
coefs = pipeline_lr.named_steps['classifier'].coef_[0]

# Ordenar los coeficientes por magnitud
sorted_indices = np.argsort(np.abs(coefs))[::-1]

# Graficar la importancia de las características
plt.figure(figsize=(10, 6))
plt.barh(feature_names[sorted_indices], coefs[sorted_indices])
plt.xlabel("Coeficiente")
plt.ylabel("Características")
plt.title("Importancia de las Variables en la Regresión Logística")
plt.show()


# Entrenar el modelo Random Forest con el pipeline
pipeline_rf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])
pipeline_rf.fit(X_train, y_train)

# Obtener la importancia de las características del modelo Random Forest
importances = pipeline_rf.named_steps['classifier'].feature_importances_

# Obtener nombres de características después del preprocesamiento
feature_names = list(preprocessor.transformers_[0][2])  # Variables numéricas
encoded_categorical_features = pipeline_rf.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
feature_names.extend(encoded_categorical_features)  # Agregar variables categóricas transformadas

# Convertir a NumPy array
feature_names = np.array(feature_names)

# Ordenar por importancia
sorted_indices = np.argsort(importances)[::-1]

# Graficar la importancia de las variables
plt.figure(figsize=(10, 6))
plt.barh(feature_names[sorted_indices], importances[sorted_indices])
plt.xlabel("Importancia de Características")
plt.ylabel("Características")
plt.title("Importancia de las Variables en Random Forest")
plt.show()

# Entrenar el modelo SVM con el pipeline
pipeline_svm = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', SVC(probability=True, random_state=42))])
pipeline_svm.fit(X_train, y_train)

# Obtener nombres de características después del preprocesamiento
feature_names = list(preprocessor.transformers_[0][2])  # Variables numéricas
encoded_categorical_features = pipeline_svm.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
feature_names.extend(encoded_categorical_features)  # Agregar variables categóricas transformadas

# Transformar los datos de prueba con el preprocesador
X_test_svm_transformed = preprocessor.transform(X_test)

# Crear un explainer de SHAP utilizando KernelExplainer
explainer_svm = shap.KernelExplainer(pipeline_svm.named_steps['classifier'].predict_proba, X_test_svm_transformed[:50])  
shap_values_svm = explainer_svm.shap_values(X_test_svm_transformed[:50])  # Calcular SHAP

# Verificar la forma de shap_values
print(f"Forma de los valores SHAP: {np.array(shap_values_svm).shape}")
print(f"Forma de X_test_svm_transformed: {X_test_svm_transformed[:50].shape}")

# Seleccionar los valores SHAP solo para la clase positiva (1) en problemas binarios
if isinstance(shap_values_svm, list):
    shap_values_svm_class1 = shap_values_svm[1]  # Elegir la clase 1 si hay múltiples matrices
else:
    shap_values_svm_class1 = shap_values_svm[..., 1]  # Seleccionar la segunda dimensión correctamente

# Asegurar que feature_names tenga la misma cantidad de elementos que X_test_svm_transformed
feature_names = feature_names[:X_test_svm_transformed.shape[1]]

# Visualizar los valores SHAP
shap.summary_plot(shap_values_svm_class1, X_test_svm_transformed[:50], feature_names=feature_names)


# Crear un explainer de SHAP con DeepExplainer
explainer_nn = shap.Explainer(model_nn, X_test_nn)

# Calcular los valores SHAP
shap_values_nn = explainer_nn(X_test_nn)

# Obtener nombres de características tras la transformación
feature_names = list(preprocessor_nn.transformers_[0][2])  # Variables numéricas
encoded_categorical_features = preprocessor_nn.transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_features)
feature_names.extend(encoded_categorical_features)  # Agregar variables categóricas transformadas

# Asegurar que feature_names tenga la misma cantidad de elementos que X_test_nn
feature_names = feature_names[:X_test_nn.shape[1]]

# Gráfico SHAP para Redes Neuronales
shap.summary_plot(shap_values_nn, X_test_nn, feature_names=feature_names)

import matplotlib.pyplot as plt
import numpy as np

# Datos de las métricas obtenidas (Clase 0)
models = ['Logistic Regression', 'Random Forest', 'SVM', 'Deep Neural Network']
precision_0 = [0.92, 0.85, 0.91, 0.86]
recall_0 = [0.68, 1.00, 0.73, 0.96]
f1_score_0 = [0.78, 0.92, 0.81, 0.91]

# AUC-ROC
auc_roc = [0.724, 0.700, 0.716, 0.680]

# Crear la posición para cada grupo en la gráfica
x = np.arange(len(models))

# Ancho de las barras
width = 0.2

# Crear la gráfica de barras
plt.figure(figsize=(10, 6))

# Métricas para la clase 0
plt.bar(x - width, precision_0, width, label='Precision', alpha=0.8)
plt.bar(x, recall_0, width, label='Recall', alpha=0.8)
plt.bar(x + width, f1_score_0, width, label='F1-Score', alpha=0.8)

# AUC-ROC como línea adicional
plt.plot(x, auc_roc, label='AUC-ROC', color='black', marker='o', linewidth=2)

# Añadir etiquetas y títulos
plt.xticks(x, models, rotation=15)
plt.xlabel("Modelos")
plt.ylabel("Métricas")
plt.title("Comparación de Modelos por Métricas")
plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.3), ncol=4)
plt.ylim(0, 1.05)  # Límite del eje Y para métricas normalizadas
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Mostrar la gráfica
plt.tight_layout()
plt.show()