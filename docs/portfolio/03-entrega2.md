<link rel="stylesheet" href="../custom.css">

# Feature Engineering y modelo base
## 2025-08-31

## Contexto
Siguiendo con el caso del titanic, se trabaja con feature engineering y un modelo base, aprendiendo el uso de baselines, preparación de datos, etc.

## Objetivos
- Investigar componentes de scikit-learn
- Aprender que es y como aplicar feature engineering
- Aprender que es el modelo base, y base line, y aplicarlos
- Evaluar nuevas metricas con la aplicación de los conceptos previamente mencionados.

## Actividades (con tiempos estimados)
- Investigación del caso — 5 min  
- Investigación de los componentes — 30 min  
- Elaboración del código — 20 min  
- Análisis de los resultados — 45 min  
- Documentación de los hallazgos — 20 min  

## Desarrollo
Inicialmente se refresco el caso brevemente, y se investigaron ciertos componentes de scikit-learn que se usarian a lo largo del codigo, a partir de la documentación oficial.

Posteriormente se empezo la elaboración del codigo en google collab, primero importando las librerias necesarias, e importando los datos del JSON de Kaggle de la misma forma que en la anterior entrega.

```python hl_lines="2 6" linenums="1"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette('deep')

from pathlib import Path
try:
    from google.colab import drive
    drive.mount('/content/drive')
    ROOT = Path('/content/drive/MyDrive/IA-UT1')
except Exception:
    ROOT = Path.cwd() / 'IA-UT1'

DATA_DIR = ROOT / 'data'
RESULTS_DIR = ROOT / 'results'
for d in (DATA_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)
print('Outputs →', ROOT)

!pip -q install kaggle
from google.colab import files
files.upload()  # Subí tu archivo kaggle.json descargado
!mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
!kaggle competitions download -c titanic -p data
!unzip -o data/titanic.zip -d data

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
```

Luego se comenzo primero con el manejo de los valores faltantes, al hacer la imputación. De Embarked para los valores faltantes se relleno con el valor mas comun, para Fare se uso la mediana, y para Age la mediana del sexo, y Pclass a la que perteneciera el record.

```python hl_lines="2 6" linenums="1"
df = train.copy()

# 🚫 PASO 1: Manejar valores faltantes (imputación)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])  # Valor más común
df['Fare'] = df['Fare'].fillna(df['Fare'].median())              # Mediana
df['Age'] = df['Age'].fillna(df.groupby(['Sex','Pclass'])['Age'].transform('median'))

# 🆕 PASO 2: Crear nuevas features útiles
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

df['Title'] = df['Name'].str.extract(',\s*([^\.]+)\.')
rare_titles = df['Title'].value_counts()[df['Title'].value_counts() < 10].index
df['Title'] = df['Title'].replace(rare_titles, 'Rare')

# 🔄 PASO 3: Preparar datos para el modelo
features = ['Pclass','Sex','Age','Fare','Embarked','FamilySize','IsAlone','Title','SibSp','Parch']
X = pd.get_dummies(df[features], drop_first=True)
y = df['Survived']

X.shape, y.shape
```

Finalmente se creo la baseline para tener una base con la que comparar nuestro modelo, ya que nos permite saber si realmente es efectivo haciendo predicciones o no. Para la baseline se uso un DummyClassifier con la estrategia de most_frequent, y para el modelo se uso regresión lineal.

Por ultimo se imprimieron las accuracy de tanto el baseline como el del modelo para compararlas, la matriz de confusion, y el reporte de clasificación.

```python hl_lines="2 6" linenums="1"
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

dummy = DummyClassifier(strategy='most_frequent', random_state=42)
dummy.fit(X_train, y_train)
baseline_pred = dummy.predict(X_test)

lr = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)
lr.fit(X_train, y_train)
pred = lr.predict(X_test)

print('Baseline acc:', accuracy_score(y_test, baseline_pred))
print('LogReg acc  :', accuracy_score(y_test, pred))

print('\nClassification report (LogReg):')
print(classification_report(y_test, pred))

print('\nConfusion matrix (LogReg):')
print(confusion_matrix(y_test, pred))
```

## Evidencias
![alt text](../assets/Entrega2Img1.png)

[Collab](https://colab.research.google.com/drive/1lEsXMWq7gRSv3R0i90mMkNZ8TXO_BH0v?usp=sharing)

## Investigación de componentes Scikit-learn

---

### LogisticRegression

***¿Qué tipo de problema resuelve?***  
Implementa un modelo de regresión lineal que nos sirve como clasificador.

***¿Qué parámetros importantes tiene?***  
Parámetros:  
`penalty, dual, tol, C, fit_intercept, intercept_scaling, class_weight, random_state, solver, max_iter, multi_class, verbose, warm_start, n_jobs, l1_ratio`

***¿Cuándo usar solver='liblinear' vs otros solvers?***  
Cuando el dataset es pequeño es una buena opción. Para datasets más grandes es mejor usar otros solvers más eficientes como **saga** y **sag**.

---

### DummyClassifier

***¿Para qué sirve exactamente?***  
Sirve como base para poder ser usado en comparaciones con otros clasificadores más complejos. Hace predicciones ignorando los inputs.

***¿Qué estrategias de baseline ofrece?***  
- most_frequent: Devuelve la clase más frecuente, como un vector.  
- prior: Devuelve la clase más frecuente, como valor de la distribución.  
- stratified: Devuelve valores aleatorios.  
- uniform: Devuelve valores de forma uniforme a partir de una lista de clases.  
- constant: Devuelve siempre el mismo valor indicado por el usuario.  

***¿Por qué es importante tener un baseline?***  
Sirve para ver si tu modelo en verdad es útil, o no sirve para nada. Lo comparas con esa baseline a partir del dummy para verificar si tu modelo hace predicciones mejores que este.

---

### train_test_split

***¿Qué hace el parámetro stratify?***  
Si no es **None**, divide los datos conservando la proporción de clases según el vector **y**.

***¿Por qué usar random_state?***  
Establece una semilla para que la partición sea reproducible en distintos experimentos.

***¿Qué porcentaje de test es recomendable?***  
Por defecto se usa un **25%** si no se indica en el parámetro **test_size** o **train_size**, pero por lo general se emplea entre **20% y 30%**.

---

### Métricas de evaluación

***¿Qué significa cada métrica en classification_report?***  
Para cada clase muestra: precisión, recall, f1-score y support.  
También incluye:  
- accuracy (global)  
- macro avg: promedio simple por clase  
- weighted avg: promedio ponderado por soporte (support)  
- micro avg (en algunos casos): promedio global de todas las clases sin ponderación  

**Definiciones:**  
- Precisión: de todo lo que predijiste como positivo, ¿cuántos fueron realmente positivos?  
- Recall: de todos los positivos reales, ¿cuántos encontraste?  
- F1-score: media armónica de precisión y recall.  
- Support: cantidad de muestras reales de cada clase.  
- Accuracy: precisión global, es decir, del total de predicciones cuántas fueron correctas.  

***¿Cómo interpretar la matriz de confusión?***  
Permite ver los verdaderos positivos, falsos positivos, falsos negativos y verdaderos negativos entre clases.

***¿Cuándo usar accuracy vs otras métricas?***  
- Si la clase está balanceada y solo importa predecir el mayor número de valores correctos, accuracy es la mejor métrica.  
- Si es importante tener en cuenta falsos positivos o falsos negativos (ejemplo: detección de enfermedades), puede ser más significativo usar precisión o recall.  
- Si lo que interesa es un balance entre la detección de positivos y negativos, lo ideal es usar f1-score.  

## Reflexión

### **Preguntas:**

**Matriz de confusión**: ¿En qué casos se equivoca más el modelo: cuando predice que una persona sobrevivió y no lo hizo, o al revés?

**Respuesta:** Que predijera que sí sobrevivió y en la realidad no sobreviviera es la celda [0,1] con valor 12, mientras que predecir que no sobrevivió pero sí sobrevivió en la realidad es la celda [1,0] con valor 21. Como 21 > 12, podemos afirmar que se equivocó más al decir que la persona no sobreviviera cuando en realidad sí sobrevivió.

---

**Clases atendidas**: ¿El modelo acierta más con los que sobrevivieron o con los que no sobrevivieron?

**Respuesta:** Tanto la precisión como el recall son mayores en la fila 0 que en la 1, como se puede ver en el Classification Report, por lo que el modelo predice mejor a los que no sobrevivieron (0.82 > 0.80 y 0.89 > 0.70).

---

**Comparación con baseline**: ¿La Regresión Logística obtiene más aciertos que el modelo que siempre predice la clase más común?

**Respuesta:** Verdadero, dado que se puede observar que el accuracy de la regresión logística es mayor que el del baseline: 0.8156 > 0.6145.

---

**Errores más importantes**: ¿Cuál de los dos tipos de error creés que es más grave para este problema?

**Respuesta:** Es más grave tener falsos negativos de personas que sobrevivieron, dado que si se usaran estos datos para rescatarlas, sería mejor captar a más personas que pudieron haber sobrevivido aunque al final no lo hicieran, que simplemente minimizar el número total de errores.

---

**Observaciones generales**: Mirando las gráficas y números, ¿qué patrones interesantes encontraste sobre la supervivencia?

**Respuesta:** Podemos observar que hay más datos de soporte de las personas que no sobrevivieron que de las que sí. Además, se suele predecir mejor la no supervivencia, hay más falsos negativos que falsos positivos, y considero que las features Age, Sex y Pclass son las más relevantes y ayudan a que el modelo pueda predecir con un accuracy del 82%.

---

**Mejoras simples**: ¿Qué nueva columna (feature) se te ocurre que podría ayudar a que el modelo acierte más?

**Respuesta:** Una posible mejora sería usar grupos de edades en vez de la edad directamente, ya que la edad por sí sola puede no capturar patrones tan claros, mientras que las agrupaciones podrían ser más útiles para predecir la supervivencia con regresión logística.

## Referencias
- https://juanfkurucz.com/ucu-ia/ut1/02-feature-modelo-base/
- https://www.kaggle.com/competitions/titanic/data
- https://scikit-learn.org/stable/user_guide.html
