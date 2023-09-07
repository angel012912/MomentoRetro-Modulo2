# Se importan las librerias necesarias
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns

"""
Se lee el archivo csv con los datos de los autos.
 - Se establecen los nombres de las columnas así como los valores que se consideran como nulos.
"""
col_names = ['mpg', 'cylinders', 'cubicinches', 'hp', 'weightlbs', 'time-to-60', 'year', 'brand']
df = pd.read_csv("cars.csv", na_values=['',' '], skiprows=1, header=None, names=col_names)

"""
Se limpian los datos:
    - En este caso únicamente es necesario limpiar los registros nulos 
    ya que tras una inspeción visual de los datos se identificó que no 
    se tienen valores categoricos o numéricos que no correspondan a los datos.
"""
df = df.dropna()


"""
Se hacen subconjuntos de los datos para separar las variables independientes de la variable dependiente.
    - Se separan las variables independientes en X y la variable dependiente en y.
    - Se separan los datos en datos de entrenamiento, validación y datos de prueba.
"""

X = df.drop('brand', axis=1)
y = df['brand']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


"""
Se crea el modelo de Random Forest con los siguientes hiper parámetros:
- n_estimators: Número de árboles en el bosque. 
    Se establece en 30, un numero que se considera no muy grande para 
    que el modelo no se llegue a sobre ajustar a los datos y que aún 
    así sea lo suficientemente grande para que el modelo tenga un buen desempeño.
- max_depth: Profundidad máxima de los árboles.
    Se establece en 4, un número bajo para que el modelo no se llegue a sobre ajustar a los datos.
- n_jobs: Número de trabajos a ejecutar en paralelo.
    Se establece en -1 para que el modelo utilice todos los procesadores disponibles.
- random_state: Semilla para el generador de números aleatorios.
    Se establece en 42, ya que tras evaluarse varias veces se identificó que genera una buena distribución de los datos.
"""
model = RandomForestClassifier(n_estimators=30, max_depth=4, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)


"""
Se evalua el desempeño del modelo con los datos de validación y de prueba: 
- Se evalua la precisión del modelo con los datos de validación y de prueba.
- Se evalua el reporte de clasificación, tanto para los datos de validación como con los de prueba, el cual muestra la precisión, el recall y el f1-score de cada clase.
- Se evalua la matriz de confusión, tanto para los datos de validación como con los de prueba, la cual muestra los valores predichos y los valores reales.
- Se grafica la matriz de confusión para los datos de validación y de prueba en un mismo plot.
- Se grafica los desempeños del modelo con los datos de train, validación y prueba en un gráfico de barras.
"""
y_pred_val = model.predict(X_val)
y_pred_test = model.predict(X_test)

print("***** Datos de validación ******")
print("Precisión de los datos de validación: ", accuracy_score(y_val, y_pred_val))
print("Reporte de clasificación de los datos de validación: \n", classification_report(y_val, y_pred_val))

print("***** Datos de prueba ******")
print("Precisión de los datos de prueba: ", accuracy_score(y_test, y_pred_test))
print("Reporte de clasificación de los datos de prueba: \n", classification_report(y_test, y_pred_test))

cm_val = confusion_matrix(y_val, y_pred_val)
cm_test = confusion_matrix(y_test, y_pred_test)

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
sns.heatmap(cm_val, annot=True, ax=ax[0], cmap=plt.cm.Blues, fmt='g')
sns.heatmap(cm_test, annot=True, ax=ax[1], cmap=plt.cm.Blues, fmt='g')
ax[0].set_title('Matriz de confusión validación')
ax[1].set_title('Matriz de confusión prueba')
ax[0].set_xlabel('Predicciones')
ax[1].set_xlabel('Predicciones')
ax[0].set_ylabel('Valores esperados')
ax[1].set_ylabel('Valores esperados')
plt.show()


plt.bar(['Entrenamiento', 'Validación', 'Prueba'], [accuracy_score(y_train, model.predict(X_train)), accuracy_score(y_val, y_pred_val), accuracy_score(y_test, y_pred_test)], color=['blue', 'orange', 'green'])
plt.title('Precisión del modelo con los datos de entrenamiento, validación y prueba')
plt.xlabel('Datos')
plt.ylabel('Precisión')
plt.show()
