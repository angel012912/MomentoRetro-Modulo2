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
    - Se separan los datos en datos de entrenamiento y datos de prueba.
"""

X = df.drop('brand', axis=1)
y = df['brand']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
Se evalua el desempeño del modelo:
- Se evalua la precisión del modelo.
- Se evalua el reporte de clasificación, el cual muestra la precisión, el recall y el f1-score de cada clase.
- Se evalua la matriz de confusión, la cual muestra los valores predichos y los valores reales.
    - Se utiliza la libreria seaborn para mostrar la matriz de confusión de una manera más visual.
"""
y_pred = model.predict(X_test)
print("Precisión: ",accuracy_score(y_test, y_pred), "\n")
print("Reporte de clasificación: \n",classification_report(y_test,y_pred))


matriz_confusion = confusion_matrix(y_test, y_pred)

data_etiquetada = pd.DataFrame(columns=["Europe","Japan", "Us"], index=["Europe","Japan", "Us"], data= matriz_confusion )

f,ax = plt.subplots(figsize=(2,2))

sns.heatmap(data_etiquetada, annot=True, cmap="Greens", fmt= '.0f',
            ax=ax,linewidths = 5, cbar = False,annot_kws={"size": 14})
plt.xlabel("Valores Predichos")
plt.xticks(size = 10)
plt.yticks(size = 10, rotation = 0)
plt.ylabel("Valores Reales")
plt.title("Matriz de confusión", size = 10)
plt.show()
