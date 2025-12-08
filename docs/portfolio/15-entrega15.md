<link rel="stylesheet" href="../custom.css">

# Agentes con LangGraph — RAG, Tools y Memoria Conversacional
## 2025-11-30

## Contexto

En esta actividad se trabajó con **LangGraph + LangChain + OpenAI** para explorar la construcción de agentes conversacionales multi-turn que integran **LLM + tools**. Se recorrió el flujo completo desde el setup básico, diseño de un estado de agente (`AgentState`), integración de RAG como tool, hasta la creación de un grafo que orquesta el reasoning del LLM con herramientas externas. El objetivo fue comprender cómo LangGraph gestiona estados, cómo vincular herramientas externas de manera reusable, y cómo mantener un historial o resumen ligero de la conversación.

## Objetivos

- Diseñar un estado de agente (`AgentState`) con historial de mensajes.  
- Construir un agente con LangGraph que:
  - Use un modelo de chat OpenAI como reasoner.
  - Llame tools externas (RAG + otras).
  - Mantenga el historial de conversación.
- Integrar RAG como tool reutilizable (retriever + LLM).  
- Agregar tools adicionales (p.ej. utilidades, servicios dummy).  
- Orquestar LLM + tools en un grafo: assistant ↔ tools con bucles.  
- Ejecutar conversaciones multi-turn y observar cómo evoluciona el estado.

## Actividades (con tiempos estimados)

- Setup e instalación de paquetes (`langgraph`, `langchain`, `faiss-cpu`) — 10 min  
- Configuración de API keys y entorno — 5 min  
- Definición del `AgentState` básico — 10 min  
- Implementación del nodo assistant y grafo mínimo — 20 min  
- Extensión del estado con summary — 15 min  
- Construcción de un RAG mini (FAISS + OpenAI embeddings) — 30 min  
- Creación de tools adicionales (hora, estado de pedido, etc.) — 20 min  
- Integración de LLM + tools con ToolNode en LangGraph — 25 min  
- Ejecución de conversaciones multi-turn — 20 min  
- Pruebas y reflexión sobre el estado y routing — 15 min

---

## Desarrollo

### Setup y modelo inicial

Se configuró el entorno instalando LangGraph, LangChain y FAISS, y se ajustaron las claves de API de OpenAI y LangChain, activando el tracing con LangSmith. Se definió un estado básico del agente para mantener el historial de mensajes y se instanció un modelo de chat determinista.

Luego se creó un nodo de asistencia que llama al modelo con todo el historial y devuelve la respuesta, y se construyó un grafo mínimo conectando inicio y fin. Se probó el agente con un mensaje de prueba, verificando que generara una respuesta correctamente.

#### Reflexión

- Diferencia entre esto y hacer llm.invoke("prompt") directo  
La diferencia es que con LangGraph no le estás tirando un prompt suelto al modelo: estás moviendo un estado que va pasando de nodo en nodo. Cada nodo agarra ese estado, lo modifica y lo vuelve a pasar. En cambio, cuando hacés llm.invoke("prompt"), no hay recorrido, no hay memoria ni pasos: solo mandás texto y recibís texto

- Estado que viaja por el grafo se ve explicitamente cuando:
  - Lo definís como AgentState.
  - Lo pasás al grafo con graph.invoke(initial_state).
  - Cada nodo lo recibe como state en def assistant_node(state).

### Extensión del estado del agente

En esta etapa se extendió el estado del agente para incluir un resumen de la conversación (summary), además del historial de mensajes. Esto permite que el agente tenga memoria ligera de lo conversado, facilitando mantener contexto sin almacenar todo el historial completo. El estado inicial se definió con la lista de mensajes vacía y el resumen en None, listo para actualizarse a medida que avanza la conversación.

#### Reflexión

- Ventaja de guardar un summary en vez de todo el historial 

Guardar un resumen en lugar de todo el historial hace que el agente sea más rápido y eficiente, porque no necesita procesar cada mensaje cada vez. Además, permite concentrarse en lo importante y mantener el contexto de la conversación sin todo el ruido de los mensajes antiguos, haciendo que la charla sea más clara y fluida.


- Información que NO deberías guardar en ese resumen por temas de privacidad 

Por motivos de privacidad, nunca se debe incluir datos sensibles de las personas, como nombres completos, direcciones, contraseñas o información financiera. El resumen debe ser útil para el agente y seguro, solo con lo necesario para continuar la conversación.

### mini RAG (Retrieval-Augmented Generation)

Construcción de un mini RAG (Retrieval-Augmented Generation) utilizando un corpus mínimo de documentos. Primero, los textos se dividieron en fragmentos con un TextSplitter para facilitar la búsqueda semántica. Luego se generaron embeddings con OpenAI y se almacenaron en un vector store FAISS, permitiendo recuperar documentos relevantes según la consulta del usuario. Finalmente, se definió una herramienta (rag_search) que realiza la búsqueda semántica y devuelve los fragmentos más relevantes como contexto, o un mensaje indicando que no se encontraron resultados.

#### Reflexión

- Si el corpus fuera mucho más grande:

Si el corpus fuera mucho más grande, habría que pensar en estrategias para no sobrecargar al modelo ni la búsqueda, probablemente debería usar un índice más eficiente, limitaría la cantidad de resultados o resumiría los documentos antes de pasarlos como contexto.

- Textos muy largos en el context: 

Devolver textos demasiado largos también puede ser un problema, porque el LLM tiene un límite de tokens y puede perder foco en lo importante, además, se vuelve más lento y caro procesar tanta información. Por eso conviene que la tool entregue solo lo relevante y conciso para que el agente pueda usarlo efectivamente.

### Tools adicionales

Se crearon tools adicionales para que el agente pueda realizar tareas más allá de la generación de texto. Se implementó get_order_status, que simula la consulta del estado de un pedido a partir de un diccionario de pedidos ficticios, y get_utc_time, que devuelve la hora actual en formato UTC. Estas herramientas permiten al agente proporcionar información dinámica y específica según la consulta del usuario, integrándose luego al grafo de LangGraph.

#### Reflexión
- Problema de este tool si la usás en producción real 
Si la usáramos en producción real, estas tools tendrían varios problemas: por un lado, FAKE_ORDERS es solo un diccionario en memoria, así que no refleja datos reales ni concurrentes, y cualquier acceso simultáneo podría generar inconsistencias. Además, no hay validación ni control de errores sofisticado, y exponer directamente funciones como get_order_status o get_utc_time podría abrir vulnerabilidades si se reciben inputs maliciosos.

- Para hacerlas más seguras y robustas habría que conectarlas a una base de datos real con control de acceso, validar y sanear los inputs, manejar errores de manera confiable, y limitar la información sensible que se devuelve, asegurando que solo se entregue lo necesario para cada usuario.

### Integración de tools al LLM

En esta sección se integraron las tools al LLM dentro de LangGraph para crear un agente capaz de decidir cuándo responder directamente o invocar herramientas externas. Se definió un nodo de asistencia (assistant_node) que procesa los mensajes con el LLM y un nodo de tools (ToolNode) que ejecuta las herramientas disponibles. Además, se implementó un enrutamiento condicional que envía el flujo hacia las tools cuando el LLM hace llamadas a herramientas, y de regreso al nodo de asistencia una vez completadas. 

Con esto se construyó un grafo completo que permite conversaciones multi-turn donde el agente puede combinar generación de texto y consultas a herramientas externas de manera coordinada.

#### Reflexión

- Ubicación del “reasoning”, su nodo 

El reasoning ahora ocurre principalmente en el nodo assistant. Es ahí donde el LLM analiza el historial de mensajes y decide si puede responder directamente o si necesita llamar a alguna tool. El nodo de tools simplemente ejecuta lo que se le pide; no decide ni razona por sí mismo.

- El diseño si tuvieras 10 tools en vez de 2-3 

Si tuvieras 10 tools en lugar de 2 o 3, el diseño debería ser más organizado: probablemente convendría tener un sistema de routing más sofisticado para decidir cuál tool llamar, o categorizar las tools por tipo de tarea para no sobrecargar al LLM con demasiadas opciones a la vez. Esto ayudaría a mantener eficiente la toma de decisiones y evitar confusiones en la elección de herramientas.

### Prueba de conversación en múltiples turnos

Se extendió la prueba de conversación a múltiples turnos, enviando un segundo mensaje mientras se conservaba el estado final del primer turno. Esto permitió al agente utilizar el historial y su base de conocimiento para generar respuestas contextualizadas. 

Además, se exploró el streaming de eventos, observando cómo se actualizan los mensajes del agente en tiempo real, lo que ayuda a verificar la ejecución de tools y la evolución del estado durante la interacción.

**Resultado**
Respuesta 2: RAG = Retrieval-Augmented Generation. En pocas palabras: es una técnica que combina un sistema de recuperación (search/recall) con un generador de lenguaje (LLM) para que las respuestas se apoyen en documentos reales en lugar de solo en la memoria del modelo.

Cómo funciona (resumen):
- Se recibe la consulta del usuario.
- Un retriever (BM25 o retriever denso con embeddings) busca documentos relevantes en una colección o vector DB.
- Se toman las top‑k evidencias y se pasan al generador (un LLM) junto con la consulta.
- El LLM produce la respuesta condicionada en esas evidencias (y la consulta), lo que ayuda a fundamentar la salida.

Componentes típicos:
- Índice/colección de documentos (base de conocimiento).
- Retriever (sparse o dense; a menudo embeddings + ANN).
- Reranker opcional.
- Generador/decoder (seq2seq LLM).
- Orquestación y lógica de fusión/aggregate (p. ej. RAG‑Sequence, RAG‑Token, o enfoques como Fusion‑in‑Decoder).

Ventajas:
- Reduce hallucinations al basar respuestas en evidencia externa.
- Permite respuestas con información actualizable sin reentrenar el modelo.
- Escalable para grandes colecciones.

Limitaciones y retos:
- Calidad depende de la recuperación (si no recupera la evidencia correcta, la respuesta puede ser errónea).
- Latencia y coste mayores (búsqueda + generación).
- Requiere gestión de índices, filtrado y control de versiones de datos.
- Aun puede “alucinar” combinando mal las evidencias o extrapolando.

Usos comunes:
- Sistemas de preguntas y respuestas sobre documentación/FAQ.
- Asistentes con acceso a bases de conocimiento internas o datos actualizados.
- Soporte técnico, legal, médico (con cuidado y validación).

Si querés, te muestro un ejemplo corto en Python (LangChain/FAISS) o un diagrama del flujo. ¿Cuál preferís?

**Resultado**
Último mensaje: human → Usá tu base de conocimiento y decime qué es RAG.
Último mensaje: ai → RAG = Retrieval-Augmented Generation. En pocas palabras: es una técnica que combina búsqueda (recuperación) de información relevante en una base de conocimientos con generación de texto por un modelo de lenguaje, para que las respuestas estén fundamentadas en documentos y no solo en la memoria del modelo.

Concepto y por qué se usa
- Problema: los LLMs pueden olvidar hechos recientes o inventar (hallucinar) respuestas.
- Solución RAG: antes de generar la respuesta, buscar fragmentos relevantes en una colección (docs, FAQ, base interna, web), y dar esos fragmentos como contexto al modelo para que los use al generar la salida.

Componentes básicos
1. Indexación: preparar y almacenar documentos (texto, PDFs, bases) en una estructura indexable.
2. Representación: convertir documentos y consultas a vectores (embeddings) o usar índices invertidos (BM25).
3. Recuperador (retriever): dado el input del usuario, devuelve los documentos o pasajes más relevantes.
4. Re-ranker (opcional): ordena o filtra los resultados recuperados para mayor precisión.
5. Generador: un LLM que recibe la consulta y los pasajes recuperados (en el prompt) y genera la respuesta final.
6. Post-procesado: citar fuentes, limitar la longitud, verificar con reglas, etc.

Tipos de retrieval
- Sparse retrieval: búsquedas por términos (BM25).
- Dense retrieval: búsqueda por similitud de embeddings (FAISS, Milvus).
- Híbrido: combina ambos para mejor cobertura.

Ventajas
- Respuestas más precisas y actualizables sin reentrenar el modelo.
- Permite auditar y citar fuentes.
- Maneja conocimiento específico de dominio o reciente.

Limitaciones y riesgos
- Calidad depende de la indexación y de la capacidad del retriever.
- Latencia mayor (recuperación + generación).
- Si los pasajes contienen errores, el modelo puede propagar o mezclar información incorrecta.
- Manejo de contexto limitado por el tamaño del prompt; hay que seleccionar bien los pasajes.

Buenas prácticas
- Limitar y seleccionar pasajes relevantes (chunking y scoring).
- Incluir instrucciones en el prompt para usar o citar solo la información recuperada.
- Re-rankeado y verificación de hechos cuando importa la exactitud.
- Actualizar el índice para mantener datos recientes.

Casos de uso
- Asistentes de atención al cliente con base de conocimientos.
- Sistemas de preguntas y respuestas sobre documentación técnica.
- Chatbots legales/medicina (con verificación y control humano).
- Resúmenes y búsqueda semántica en grandes colecciones.

Si querés, te muestro un flujo simple paso a paso o un ejemplo de prompt/arquitectura con herramientas comunes (FAISS, embeddings, LLM). ¿Querés eso?

#### Reflexión

- El agente está llamando rag_search vs get_order_status 

Se puede ver claramente cuándo el agente está llamando a rag_search o a get_order_status porque el LLM genera una tool call en su mensaje. Esa llamada indica que no está respondiendo directamente, sino que necesita ejecutar una herramienta para obtener información.

- Prompts para que use tools “con criterio” 

Para que el modelo use las tools con criterio, conviene darle prompts claros que expliquen qué hace cada tool, cuándo conviene usarla y qué tipo de respuesta se espera. De esta manera, el LLM puede decidir de forma más inteligente si necesita recurrir a una tool o si puede responder directamente con lo que ya sabe.

### Implementación de nodo de memoria ligera

Se implementó un nodo de memoria ligera que genera un resumen de la conversación hasta el momento, consolidando los acuerdos o puntos clave entre el usuario y el asistente. Este nodo utiliza un LLM para procesar el historial de mensajes y el resumen previo, produciendo un resumen conciso en formato de puntos.

Además, se definió una lógica condicional para actualizar la memoria cada ciertos turnos de la conversación, integrando el nodo de memoria dentro del grafo de manera que el flujo vuelva al nodo de asistencia tras cada actualización. Esto permite que el agente mantenga contexto de manera eficiente sin depender únicamente del historial completo.

#### Reflexión

- Actualizar el summary 

Se debe actualizar el summary cada cierto número de turnos o cuando la conversación haya cambiado de tema de manera significativa, de modo que refleje el contexto relevante sin ser redundante ni demasiado frecuente.

- Info a excluir del summary 

En el resumen no incluiría información sensible o privada de los usuarios, como datos personales, contraseñas o detalles financieros; solo debería concentrarse en lo importante para que el agente recuerde el hilo de la conversación y pueda dar respuestas coherentes.

### Interfaz interactiva

Se desarrolló una interfaz interactiva con Gradio para probar el agente sin necesidad de modificar el código en cada turno. La UI permite enviar preguntas, visualizar el historial de la conversación y registrar qué tools fueron invocadas en cada respuesta. Al ejecutar el grafo, los mensajes del usuario se agregan al estado del agente y se genera la respuesta correspondiente, mientras que las llamadas a herramientas se muestran en un panel aparte. Esta implementación facilita la experimentación y la observación del comportamiento del agente en tiempo real.

### Prueba integradora final

En esta última sección se construyó un mini-agente de soporte integrando RAG y herramientas adicionales para consultas corporativas. Se definió un corpus simple con información de la empresa, que se dividió en fragmentos y se indexó en FAISS para búsquedas semánticas. Se implementaron dos tools principales: buscar_info para recuperar información específica del corpus y estado_sistema para reportar el estado del sistema con la hora actual. Luego, se integraron estas tools en un LLM usando LangGraph, creando un grafo donde el agente puede decidir cuándo responder directamente o invocar herramientas según la pregunta del usuario. 

Finalmente, se realizaron pruebas de interacción mostrando distintos escenarios: preguntas generales, consultas que requieren RAG y consultas múltiples combinando información de diferentes tools, validando el correcto flujo de llamadas y la generación de respuestas coherentes. 

Una interfaz de chat interactiva con Gradio para el mini-agente de soporte, permitiendo que los usuarios realicen preguntas sobre horarios, planes o estado del sistema de manera directa. La función principal reconstruye el estado del agente a partir del historial de la conversación y agrega cada nuevo mensaje antes de invocar el grafo. La interfaz también muestra cuándo se usan tools en la respuesta, mantiene el historial de la conversación y permite limpiar el chat con un botón. Esta implementación facilita la prueba del agente en escenarios realistas y permite observar de manera inmediata cómo combina respuestas generadas por el LLM con información obtenida de herramientas externas.

**Resultado**
=== PRUEBA 1: Pregunta general (sin tools) ===
Respuesta: ¡Hola! Estoy aquí y listo para ayudarte. ¿En qué puedo asistirte hoy?
Tools usadas: Ninguna (respuesta directa)
---
=== PRUEBA 2: Pregunta que necesita RAG ===
Respuesta: El horario de atención es de lunes a viernes de 9:00 a 18:00. Si necesitas soporte técnico, puedes contactarlos por email a soporte@empresa.com.
---
=== PRUEBA 3: Consulta múltiple ===
Respuesta: ### Planes de la Empresa
- **Plan Starter**: $50
- **Plan Pro**: $100
- **Plan Enterprise**: $200
- Para soporte técnico, puedes contactarlos por email: soporte@empresa.com.

### Estado del Sistema
- El sistema está operativo normalmente.
- Hora actual: 2025-11-30 18:22:09.

## Evidencias
- [Collab](https://colab.research.google.com/drive/1lyYC0JWZxmRKrGqxBWT21f-fCmAGBspF?usp=sharing)

## Referencias

- https://juanfkurucz.com/ucu-ia/ut4/15-agents/#parte-7-interfaz-gradio-para-probar-el-agente
