<link rel="stylesheet" href="./custom.css">

# Introducción al entorno de Google Cloud
## 2025-12-01

## Contexto
Este laboratorio introductorio se realizó en la plataforma Google Cloud Skills Boost, utilizando un entorno temporal provisto por Qwiklabs. El objetivo fue familiarizarse con la consola de Google Cloud, comprender el funcionamiento del acceso con credenciales temporales, y reconocer elementos clave como proyectos, paneles de navegación y roles de IAM.

## Objetivos
- Familiarizarse con el entorno temporal de Google Cloud Skills Boost.
- Comprender el funcionamiento del botón Start Lab y el temporizador.
- Navegar por la consola de Google Cloud y reconocer sus elementos principales.
- Entender el concepto de Proyecto como unidad organizativa de recursos.
- Explorar el menú de navegación y la ubicación de servicios clave.
- Identificar los roles básicos de IAM (Viewer, Editor, Owner) y su alcance.
- Diferenciar entre el proyecto asignado para el laboratorio y el proyecto compartido Qwiklabs Resources.

## Actividades (con tiempos estimados)
- Activación del laboratorio con Start Lab — 5 min
- Exploración de la consola y reconocimiento de la interfaz — 15 min
- Identificación del proyecto asignado y del proyecto Qwiklabs Resources — 10 min
- Navegación por el menú principal (Compute, Storage, Networking, IAM, etc.) — 15 min
- Revisión de roles de IAM asignados al usuario temporal — 10 min
- Reflexión sobre la gestión del tiempo y el acceso temporal — 5 min

## Desarrollo
### Activación y entorno temporal
El laboratorio se inicia con el botón Start Lab, el cual genera automáticamente:

- Credenciales temporales para un usuario específico (ej. student-xx-xxxx@qwiklabs.net).
- Un proyecto de Google Cloud aislado para realizar las tareas.
- Un temporizador que indica la duración restante del acceso (usualmente 1-2 horas).

Al finalizar el tiempo, el acceso se revoca y todos los recursos creados en el proyecto asignado dejan de estar disponibles. Algunos laboratorios también incluyen un sistema de tracking que valida automáticamente el cumplimiento de ciertas tareas.

#### Reflexión
Importancia del temporizador: Este sistema obliga a trabajar de manera ordenada y eficiente, similar a las ventanas de mantenimiento controladas en entornos reales. No prestar atención al tiempo o a las credenciales suele ser la causa principal de errores en estos labs.

### Navegación por la consola de Google Cloud
La primera actividad práctica consistió en explorar los elementos del panel izquierdo de la consola, donde se encuentra:
- Información del usuario y cambio de proyecto.
- Acceso rápido a todos los servicios de Google Cloud (agrupados por categorías: Compute, Storage, Networking, etc.).
- Paneles de administración como IAM, Billing y Logs.

El usuario asignado es una identidad de IAM con permisos específicos y restringidos, aplicando el principio de privilegio mínimo incluso en un entorno de entrenamiento.

#### Reflexión
Seguridad desde el inicio: Que Google aplique permisos restringidos en un entorno de práctica refuerza la mentalidad de "mínimo privilegio necesario", una buena práctica fundamental en la administración de la nube.

### El Proyecto en Google Cloud
Se explicó que un Proyecto en Google Cloud actúa como:
- La entidad organizadora principal para todos los recursos (VMs, buckets, redes, etc.).
- La unidad de facturación.
- El contenedor donde se aplican políticas, roles y permisos de IAM.

Además, se destacó la existencia del proyecto Qwiklabs Resources, el cual es compartido en modo de solo lectura por todos los usuarios y no puede ser modificado ni eliminado. Es crucial entender que las tareas del laboratorio deben realizarse en el proyecto asignado individualmente, no en este proyecto compartido.

#### Reflexión
Aislamiento y consistencia: Separar los recursos de práctica de un proyecto compartido evita conflictos entre usuarios y asegura un entorno de aprendizaje consistente y reproducible para todos.

### Roles básicos de IAM
Se revisaron los tres roles primarios a nivel de proyecto:
- Viewer (roles/viewer): Permiso de solo lectura sobre la mayoría de los recursos.
- Editor (roles/editor): Incluye permisos de Viewer y permite modificar recursos existentes.
- Owner (roles/owner): Control total, incluyendo la capacidad de gestionar roles y permisos para otros.

Estos roles se aplican a todo el proyecto y afectan a todos los servicios dentro de él.

#### Reflexión
Fundamental para entornos reales: Entender la granularidad y el poder de estos roles es esencial, ya que en organizaciones reales una mala asignación puede generar riesgos de seguridad o costos innecesarios.

### Habilitación de APIs y Servicios
Para utilizar cualquier servicio de Google Cloud (como Dialogflow, Compute Engine, etc.) en un proyecto, primero se debe habilitar su API correspondiente.

Procedimiento realizado:
1. En el menú de navegación, fui a APIs y servicios > Biblioteca.
2. Busqué "Dialogflow" y seleccioné la API.
3. Hice clic en "Habilitar".
4. Verifiqué que apareciera como habilitada en el listado.
5. Probé acceder a la documentación de la API desde la misma consola.

**Resultado**
![alt text](<../assets/entrega16Img.jpeg>)

#### Reflexión 
Aunque en los labs las APIs clave vienen pre-habilitadas, en proyectos reales este es un paso esencial. Cada API habilitada puede tener costos asociados y debe gestionarse según las necesidades específicas de la aplicación.

### Conclusiones del laboratorio
Este laboratorio introductorio sentó las bases conceptuales y prácticas para trabajar en Google Cloud:
- Entorno Temporal: Se comprendió la dinámica de acceso limitado y la importancia de gestionar el tiempo.
- Estructura Organizativa: Se asimiló el concepto central del Proyecto.
- Navegación: Se ganó fluidez para moverse por la consola y localizar servicios.
- Gestión de Accesos: Se introdujo el modelo de permisos a través de IAM.

#### Reflexión final
Cimiento para lo complejo: Aunque sencillo, este laboratorio es fundamental. Un entendimiento claro de la consola y la estructura de permisos previene una gran cantidad de problemas y confusiones en laboratorios más avanzados que implican la creación y configuración de recursos.

## Referencias

- https://www.skills.google/focuses/2794?catalog_rank=%7B%22rank%22%3A3%2C%22num_filters%22%3A2%2C%22has_search%22%3Atrue%7D&parent=catalog&search_id=60924676

