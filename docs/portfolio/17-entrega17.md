<link rel="stylesheet" href="../custom.css">

# Vertex AI Pipelines: Qwik Start
## 2025-12-01

## Contexto

En este laboratorio trabajé con **Vertex AI Pipelines**, la herramienta de Google Cloud para armar y automatizar flujos de machine learning. La idea general fue poner en práctica cómo se orquestan procesos de ML de punta a punta: desde preparar datos hasta entrenar y desplegar modelos, combinando componentes propios y los que ya ofrece Google.

## Objetivos

- Crear y ejecutar un pipeline básico de tres pasos usando el SDK de Kubeflow Pipelines (KFP).
- Construir un pipeline completo de clasificación tabular usando AutoML.
- Aprovechar componentes prearmados del paquete google_cloud_pipeline_components.
- Agregar lógica condicional para decidir automáticamente si un modelo se despliega o no.
- Comparar ejecuciones y revisar métricas entre distintos runs.

## Actividades (con tiempos estimados)

- Configuración inicial en Vertex AI Workbench — 10 min  
- Instalación de dependencias — 15 min  
- Pipeline introductorio “hello-world” — 25 min  
- Pipeline de clasificación con AutoML — 30 min  
- Ejecución y monitoreo — 30 min  
- Análisis de resultados — 10 min  

---

## Desarrollo

### Configuración del entorno y bibliotecas

Arranqué creando una instancia nueva de JupyterLab desde Vertex AI Workbench. Instalé:

- kfp (para definir pipelines)  
- google-cloud-pipeline-components  
- google-cloud-aiplatform  

Después del restart del kernel configuré las variables principales como **PROJECT_ID** y **BUCKET_NAME**, que usé para guardar artefactos y outputs.

#### Reflexión

- Dependencias: Aparecieron algunos warnings de compatibilidad. Nada grave, pero en un proyecto real lo solucionaría con entornos más controlados (virtualenv/conda o imágenes Docker).
- Bucket: Tener el bucket bien definido desde el principio simplifica todo, porque los pipelines van a dejar ahí prácticamente todo lo que generan.

---

### Pipeline introductorio "hello-world"

El primer ejercicio fue armar un pipeline pequeño para entender cómo KFP convierte funciones Python en componentes.

Los tres pasos fueron:

1. product_name — toma un texto y lo devuelve.  
2. emoji — convierte una palabra tipo “sparkles” en ✨.  
3. build_sentence — junta las salidas y arma una frase.

Lo importante acá fue ver cómo el decorador @component envuelve cada función en un contenedor independiente, con sus propias dependencias. Esto facilita mucho cuando cada paso necesita un entorno distinto.

Compilé el pipeline a JSON, lo subí a Vertex AI Pipelines y vi el grafo ejecutándose en tiempo real. El resultado final fue: **"Vertex AI Pipelines is ✨"**.

**Resultado**  
![alt text](<../assets/entrega17Img1.jpeg>)

#### Reflexión

- Componentización: Convertir funciones sueltas en componentes reutilizables hace que los pipelines sean más mantenibles.
- UI de Vertex AI: Ver el grafo correr paso por paso ayuda muchísimo cuando algo falla. Es más visual que leer logs crudos.

---

### Pipeline de ML completo con AutoML

El segundo pipeline ya fue de verdad “realista”. Entrené un modelo de clasificación usando AutoML y armé un flujo completo:

1. Carga de datos (TabularDatasetCreateOp)  
   Importé datos desde BigQuery.

2. Entrenamiento con AutoML (AutoMLTabularTrainingJobRunOp)  
   Definí transformaciones, presupuesto de entrenamiento y métricas objetivo.

3. Evaluación del modelo (componente propio)  
   - Extrajo métricas.  
   - Mostró ROC y matriz de confusión.  
   - Comparó el AUC con un umbral (0.95).  

4. Despliegue condicional  
   Si el AUC superaba el umbral, el modelo se desplegaba automáticamente con ModelDeployOp.

Mientras AutoML entrenaba, pude ver cómo Vertex AI manejaba todas las transformaciones y el tuning sin que yo escribiera casi nada de código manual.

![alt text](<../assets/entrega17Img2.jpeg>)

#### Reflexión

- AutoML: Para prototipar es excelente. Te permite validar ideas sin invertir tiempo en feature engineering o tuning profundo.
- Condicionales: El dsl.Condition() le da inteligencia al pipeline. Podría usarse para automatizar reentrenamientos, alertas o decisiones de despliegue.
- Linaje: Poder rastrear de qué dataset salió cada modelo o experimento es súper útil cuando se auditan resultados o se necesita reproducibilidad.

---

## Resultados y análisis

### Pipeline "hello-world"
- Tardó unos 6 minutos.  
- Generó el output esperado: “Vertex AI Pipelines is ✨”.  
- Guardó sus artefactos en Cloud Storage.

### Pipeline de AutoML
- El dataset desde BigQuery cargó sin problemas.  
- El modelo alcanzó un AUC de ~0.98 (pasó el umbral).  
- El despliegue se ejecutó automáticamente.  
- Las métricas se visualizaron correctamente:
  - Curva ROC casi ideal.  
  - Matriz de confusión bastante equilibrada entre las 7 clases.  

---

## Observaciones finales

1. Infraestructura como código: Con KFP todo queda versionado y reproducible.  
2. Control de costos: Los pipelines dejan claro qué recursos consume cada paso.  
3. Monitoreo: Vertex AI ofrece dashboards y linaje sin hacer nada extra.  
4. Escalabilidad: El mismo pipeline funciona igual con datos chicos o gigantes, solo cambian los recursos que asignás.

---

## Conclusión

Vertex AI Pipelines resultó ser una plataforma muy completa para orquestar flujos de ML sin meterse tanto con infraestructura. Lo que más me sorprendió fue lo rápido que se pasa de funciones simples a pipelines complejos y productivos, especialmente usando los componentes prearmados de Google.

Si tuviera que aplicarlo en un proyecto real, seguramente arrancaría con AutoML y los componentes estándar, y recién después agregaría pasos personalizados cuando realmente los necesite. Menos código propio significa menos puntos de mantenimiento —y Vertex AI ya resuelve una gran parte del trabajo pesado.
