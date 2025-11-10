---
title: "Comparación de CNN desde cero vs Transfer Learning"
date: 2025-11-09
---

## Contexto

En esta actividad se trabajó con el dataset *CIFAR-10*, un conjunto de 60,000 imágenes en 10 clases, ampliamente utilizado para evaluar modelos de visión por computadora.

El objetivo principal fue comparar el desempeño entre una *CNN construida desde cero* y un modelo utilizando *Transfer Learning* con MobileNetV2 preentrenado en ImageNet

Esto permite analizar cuándo y por qué puede resultar ventajoso transferir conocimiento previo frente a entrenar un modelo desde cero.

## Objetivos

- Cargar y preparar el dataset CIFAR-10
- Implementar una red convolucional básica
- Implementar un modelo con Transfer Learning congelando capas
- Entrenar ambos modelos bajo las mismas condiciones
- Realizar comparación basada en accuracy, pérdida y overfitting
- Interpretar resultados

## Actividades (con tiempos estimados)

- Elaboración inicial y entendimiento del dataset — 20 min  
- Exploración de posibilidades de configuración — 20 min  
- Elaboración de código final — 50 min  
- Documentación de los hallazgos — 40 min  

## Desarrollo

### Preparación de datos

- Se normalizaron las imágenes a escala [0, 1]
- Se codificaron los labels en formato one-hot
- Se utilizaron batchs de tamaño *128*

### Modelo 1: CNN Simple (desde cero)

La arquitectura incluyó:

- 2 bloques Convolución + ReLU + MaxPooling
- Capa densa de 512 neuronas + ReLU
- Capa final softmax para clasificación

Optimizador: Adam (lr = 0.001) 
Pérdida: categorical_crossentropy

### Modelo 2: Transfer Learning

Se utilizó *MobileNetV2 (weights = ImageNet, include_top = False)*

- Se congelaron todas sus capas
- Se agregó capa Flatten + capa Dense final de clasificación
- Se entrenó únicamente la parte añadida

Optimizador: Adam (lr = 0.001)
Pérdida: categorical_crossentropy

## Resultados

![alt text](../assets/Entrega9Img1.png)

### Accuracy Final en Test

| Modelo | Test Accuracy |
|-------|--------------|
| CNN Simple | *70.17%* |
| Transfer Learning (MobileNetV2) | *31.32%* |

La CNN simple obtuvo un rendimiento *significativamente superior* al modelo con Transfer Learning en este escenario.

### Análisis

MobileNetV2 fue entrenado originalmente para imágenes a 224×224, mientras que CIFAR-10 usa 32×32.

Al congelar todas las capas, el modelo no reajustó sus filtros a las características de CIFAR-10. En consecuencia, el modelo actuó casi como un extractor de características poco útil para este tamaño y dominio de imagen.

Por otro lado:

- La *CNN simple sí aprendió específicamente para CIFAR-10*
- Esto permitió obtener patrones más relevantes para este dataset.

### Overfitting

Podría observarse en las curvas de entrenamiento (según gráficos generados) la diferencia entre precisión de entrenamiento y validación:

- CNN Simple gap ≈ moderado
- Transfer Learning gap ≈ bajo pero con bajo accuracy general


Parece que el modelo transferido no sobreajustó, simplemente no aprendió patrones relevantes.

## Conclusión

- El Transfer Learning *no siempre garantiza mejor rendimiento*, especialmente cuando el modelo base fue entrenado en un dominio visual muy distinto, las imágenes tienen resoluciones mucho más pequeñas y no se realiza fine-tuning

La *CNN simple fue claramente la mejor opción, alcanzando *~70% de accuracy**, lo cual es razonable para una arquitectura básica en CIFAR-10.

- Si se quisiera mejorar el Transfer Learning, sería recomendable:
  - *Aumentar el tamaño de entrada a 96×96 o 128×128*
  - *Descongelar parcialmente capas finales* (fine-tuning)
  - *Utilizar data augmentation*

## Evidencias
- [Collab](https://colab.research.google.com/drive/1a2QrUS58Jl4UjvGx1Lc3ZwyREpr044Ec?usp=sharing)

## Referencias

- https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2
- https://www.tensorflow.org/tutorials/images/transfer_learning
- https://juanfkurucz.com/ucu-ia/ut3/09-cnn-transfer-learning-assignment/
- https://www.cs.toronto.edu/~kriz/cifar.html