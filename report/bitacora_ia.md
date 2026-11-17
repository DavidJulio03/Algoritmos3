# Bitácora de Inteligencia Artificial (Anexo Obligatorio)

- **Asignatura:** Algoritmos y Arquitecturas Computacionales de Alto Rendimiento
- **Semestre:** 2026-2
- **Institución:** Universidad Industrial de Santander (UIS)
- **Documento de Referencia:** `Ev3_2026-2.pdf`
- **Fecha de Elaboración:** Noviembre 2026

---

## 1. Herramientas Utilizadas

| Herramienta | Versión / Modelo | Propósito Principal |
|---|---|---|
| **Antigravity AI (Google DeepMind)** | Gemini 3.8 Flash (High) | Asistente de arquitectura de software científico, depuración de scripts Numba y estructuración de código de alto rendimiento. |
| **Python** | 3.12.10 (AMD64) | Entorno base de desarrollo y ejecución de algoritmos numéricos. |
| **Numba** | 0.68.1 | Compilación JIT de funciones en CPU (`parallel=True`) y definición de kernels CUDA (`numba.cuda`). |
| **llvmlite** | 0.50.0 | Capa de enlace hacia la infraestructura de compilación LLVM utilizada por Numba. |
| **NumPy** | 2.5.5 | Operaciones matemáticas vectoriales y manejo de arreglos numéricos en memoria. |
| **Pytest** | 9.2.0 | Suite automatizada de pruebas unitarias y de integración para CPU y GPU. |
| **Matplotlib** | 3.11.2 | Generación de figuras científicas de convergencia y speedup en escala log-log y semilog. |
| **Pandas** | 3.1.1 | Estructuración, consolidación y exportación de mediciones experimentales en formato CSV. |

---

## 2. Registro Cronológico de Prompts Enviados

A continuación se transcribe el texto íntegro de los prompts recibidos y procesados durante el desarrollo del proyecto, sin abreviaciones ni reconstrucciones de memoria.

### Prompt 1: Desarrollo Inicial del Proyecto (Sesión Principal)

````text

# DESARROLLO DE EVALUACIÓN — INTEGRACIÓN DE MONTE CARLO Y COMPUTACIÓN DE ALTO RENDIMIENTO

## 1. Rol y objetivo

Actúa como un ingeniero de software especializado en computación científica, algoritmos paralelos, arquitectura de computadores, Python, Numba y programación CUDA.

Tu objetivo es desarrollar completamente el trabajo académico descrito en el documento `Ev3_2026-2.pdf`, perteneciente a la asignatura **Algoritmos y Arquitecturas Computacionales de Alto Rendimiento de la Universidad Industrial de Santander**.

Debes cumplir rigurosamente todos los requisitos del documento, desarrollar los tres niveles de implementación, realizar los experimentos computacionales, generar los resultados y preparar los entregables solicitados.

Sin embargo, existe una condición fundamental: el proyecto debe ser comprensible para un estudiante universitario que tendrá que explicar personalmente el funcionamiento de cada implementación, especialmente el kernel CUDA.

No quiero una solución que simplemente funcione. Quiero una solución correcta, verificable, organizada, reproducible y que pueda comprender y defender.

## 2. Antes de comenzar: análisis del documento y del entorno

Primero:

1. Lee completamente `Ev3_2026-2.pdf`.
2. Extrae los requisitos de cada nivel, los entregables y los criterios de evaluación.
3. Inspecciona el workspace actual antes de crear, modificar o eliminar archivos.
4. Identifica el sistema operativo, versión de Python, CPU, número de núcleos e hilos disponibles, RAM y disponibilidad de GPU CUDA.
5. Comprueba las versiones de las bibliotecas instaladas y las posibilidades reales de ejecutar Numba CUDA.
6. Determina si el proyecto se ejecutará localmente o si es necesario utilizar Google Colab para el nivel 3.

No asumas que existe una GPU compatible con CUDA. Comprueba su disponibilidad.

Si alguna limitación del entorno impide completar una actividad, explícalo y propón una alternativa viable sin presentar resultados simulados como resultados experimentales.

Antes de modificar archivos existentes, determina si contienen trabajo que deba conservarse.

## 3. Problema matemático

El problema consiste en calcular el momento de inercia de un disco delgado de radio R, centrado en el origen, cuya densidad superficial está definida por:

ρ(x,y) = ρ₀(1 + r/R)

Donde:

r = √(x² + y²)

El momento de inercia respecto al eje z es:

I = ∬D ρ(x,y)r² dA

La solución analítica de referencia es:

I_exacto = (9/10)πρ₀R⁴

Utiliza:

ρ₀ = 1
R = 1

Por tanto, el resultado exacto que debe utilizarse para validar las implementaciones es:

I_exacto = 0.9π

La aproximación de Monte Carlo indicada en el documento consiste en muestrear uniformemente N puntos dentro del cuadrado [-R,R] × [-R,R], utilizar una función indicadora para descartar los puntos que estén fuera del disco y calcular:

I ≈ A · (1/N) Σ ρ(xᵢ,yᵢ)rᵢ² · 1[rᵢ ≤ R]

Donde:

A = 4R²

Respeta esta formulación. No cambies el dominio de muestreo ni utilices otro estimador sin justificarlo y comprobar que sigue siendo equivalente al método solicitado.

## 4. Arquitectura del proyecto

Organiza el proyecto de manera modular, clara y profesional.

Como punto de partida, utiliza una estructura similar a esta:

```text
monte-carlo-inertia/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── monte_carlo_sequential.py
│   ├── monte_carlo_cpu.py
│   ├── monte_carlo_gpu.py
│   ├── benchmarks.py
│   ├── validation.py
│   └── utils.py
│
├── results/
│   ├── raw/
│   ├── processed/
│   ├── tables/
│   └── figures/
│
├── notebooks/
│
├── report/
│   ├── informe.md
│   └── bitacora_ia.md
│
└── tests/
    ├── test_sequential.py
    ├── test_cpu.py
    └── test_gpu.py
```

Puedes modificar esta estructura si existe una razón técnica válida.

Evita la duplicación innecesaria de código. Centraliza las constantes, funciones auxiliares y configuraciones compartidas.

No introduzcas abstracciones complejas que dificulten la comprensión de un proyecto académico de esta escala.

## 5. Nivel 1 — Implementación secuencial

Desarrolla la implementación secuencial en Python.

Requisitos:

- Implementar el estimador de Monte Carlo solicitado.
- Generar puntos aleatorios uniformemente distribuidos dentro del cuadrado.
- Aplicar correctamente la función indicadora del disco.
- Calcular el momento de inercia aproximado.
- Comparar el resultado con el valor analítico.
- Calcular el error absoluto.

Utiliza los siguientes tamaños de muestra:

N = 10³, 10⁴, 10⁵, 10⁶ y 10⁷.

Calcula:

Error absoluto = |I_estimado - I_exacto|

Genera una gráfica del error absoluto en función de N utilizando escala log-log.

Incluye una recta de referencia proporcional a:

N^(-1/2)

Analiza si los resultados muestran el comportamiento estadístico esperado.

Ten presente que Monte Carlo tiene variabilidad aleatoria: no exijas que cada punto siga perfectamente la recta de referencia.

Utiliza semillas controladas para permitir reproducibilidad.

Explica cómo influye el número de muestras en la precisión y por qué se espera una convergencia estadística del orden de 1/√N.

## 6. Nivel 2 — Paralelización en CPU con Numba

Implementa una versión paralela utilizando:

- `numba.njit`
- `parallel=True`
- `numba.prange`

El cálculo debe estar realmente paralelizado y no limitarse a ejecutar una función secuencial dentro de una función decorada.

Requisitos:

1. Implementar el cálculo paralelo.
2. Garantizar una reducción correcta de las contribuciones parciales.
3. Evitar condiciones de carrera.
4. Mantener la precisión numérica.
5. Comparar los resultados con la versión secuencial.

Realiza benchmarks para:

N = 10⁴, 10⁵, 10⁶, 10⁷ y 10⁸.

Mide los tiempos de ejecución de ambas versiones.

IMPORTANTE: Numba utiliza compilación JIT. Debes ejecutar previamente las funciones necesarias para que la compilación no contamine las mediciones.

Consideraciones adicionales:

- Realiza calentamiento (warm-up).
- No incluyas la primera compilación en los tiempos.
- Realiza varias repeticiones por configuración.
- Utiliza una estadística representativa, preferiblemente la mediana.
- Mantén condiciones de medición comparables.
- Documenta el número de núcleos e hilos de CPU disponibles.
- Evita comparar mediciones realizadas con configuraciones de paralelismo diferentes sin documentarlo.

Calcula:

Speedup = T_secuencial / T_paralelo

Genera una tabla y una gráfica del speedup en función de N.

Explica por qué para tamaños pequeños el paralelismo puede ser más lento que la ejecución secuencial.

No presupongas que la versión paralela siempre será más rápida.

## 7. Nivel 3 — Paralelización en GPU con Numba CUDA

Implementa el algoritmo utilizando `numba.cuda`.

Este nivel requiere especial atención porque es uno de los componentes más importantes de la evaluación.

Debes desarrollar un kernel CUDA que:

1. Genere números aleatorios utilizando los mecanismos compatibles con CUDA de Numba.
2. Utilice `numba.cuda.random` y generadores `xoroshiro128p`.
3. Asigne un estado de generación aleatoria a cada hilo.
4. Genere las coordenadas de los puntos.
5. Calcule las contribuciones al momento de inercia.
6. Aplique la función indicadora.
7. Acumule resultados parciales.
8. Permita realizar una reducción para obtener el resultado final.

No utilices `np.random` dentro del kernel, porque no es compatible con la ejecución CUDA.

### Distribución del trabajo

Implementa una distribución de trabajo comprensible y eficiente entre:

- Bloques.
- Hilos por bloque.
- Hilos totales.
- Muestras procesadas por hilo.

Utiliza índices globales correctamente para evitar que varios hilos procesen accidentalmente las mismas muestras o que algunas queden sin procesar.

### Reducción

Diseña un mecanismo correcto para acumular las contribuciones parciales y obtener el resultado global.

Explica claramente cómo se pasa de las contribuciones individuales de los hilos a la suma final.

Evita transferencias innecesarias entre CPU y GPU.

### Decisiones técnicas

Documenta y justifica:

- Número de hilos por bloque.
- Número de bloques.
- Número de muestras por hilo.
- Tipo de dato utilizado: float32 o float64.
- Generación de números aleatorios.
- Estrategia de reducción.
- Transferencias de memoria.
- Consideraciones de precisión.

No elijas configuraciones arbitrariamente: utiliza pruebas y mediciones para determinar configuraciones razonables.

### Comparación

Compara los resultados de GPU con los de CPU.

Verifica que las diferencias sean compatibles con el error estadístico de Monte Carlo y con las diferencias de precisión numérica.

Mide los tiempos de ejecución y agrega la versión GPU a las tablas y gráficas de rendimiento.

Distingue entre:

- Tiempo de generación o preparación de estados aleatorios.
- Tiempo de ejecución del kernel.
- Tiempo de transferencia de datos.
- Tiempo total de ejecución.

Explica qué medición utilizas para la comparación y por qué.

Si utilizas eventos CUDA para medir el kernel, asegúrate de sincronizar correctamente la ejecución.

No presentes tiempos de kernel como si fueran tiempos totales cuando omitan operaciones relevantes.

Si el equipo local no dispone de una GPU CUDA compatible, prepara una implementación ejecutable en Google Colab y documenta los pasos necesarios para utilizar una GPU allí.

No inventes benchmarks de GPU.

## 8. Validación científica y pruebas

Implementa pruebas que permitan detectar errores matemáticos, algorítmicos y de ejecución.

Verifica:

- Que las constantes sean correctas.
- Que el área utilizada sea 4R².
- Que la función indicadora descarte correctamente puntos fuera del disco.
- Que los resultados sean físicamente y matemáticamente razonables.
- Que la implementación secuencial sea correcta.
- Que la versión paralela produzca resultados consistentes.
- Que el kernel CUDA procese las muestras correctamente.
- Que no existan errores en la reducción.
- Que las diferencias entre implementaciones sean razonables.

Distingue entre reproducibilidad exacta y reproducibilidad estadística. Las implementaciones CPU y GPU no tienen que producir exactamente la misma secuencia aleatoria.

No establezcas tolerancias arbitrarias sin justificar su significado.

Ejecuta las pruebas disponibles y registra sus resultados.

## 9. Experimentos y resultados

Los resultados experimentales deben generarse ejecutando realmente el código.

Nunca:

- Inventes tiempos de ejecución.
- Inventes resultados numéricos.
- Inventes características del hardware.
- Rellenes tablas con valores estimados.
- Presentes resultados teóricos como si fueran mediciones.
- Afirmes que una GPU fue utilizada si no se ejecutó el kernel.

Guarda los resultados obtenidos en archivos CSV o formatos estructurados equivalentes.

Organiza las mediciones para poder reproducir las gráficas sin volver a ejecutar todos los experimentos.

Registra al menos:

- Fecha de ejecución.
- Hardware.
- Sistema operativo.
- Versiones de Python y bibliotecas.
- Tamaño de muestra.
- Implementación utilizada.
- Semilla, cuando corresponda.
- Tiempo medido.
- Resultado obtenido.
- Error absoluto.
- Speedup.

Si alguna prueba no puede ejecutarse, deja explícitamente identificado ese resultado como pendiente.

## 10. Informe académico

Prepara un informe corto que respete el límite de tres páginas indicado en la evaluación, sin contar la bitácora de IA.

El informe debe contener:

1. Introducción y objetivo.
2. Descripción del método de Monte Carlo.
3. Implementación secuencial y validación.
4. Implementación paralela en CPU.
5. Implementación en GPU.
6. Gráfica de convergencia del error.
7. Tabla comparativa de tiempos.
8. Gráfica de speedup.
9. Análisis de resultados.
10. Conclusiones.

Prioriza los resultados experimentales y evita explicaciones demasiado extensas.

Las gráficas deben tener:

- Título.
- Ejes correctamente etiquetados.
- Unidades cuando correspondan.
- Leyendas.
- Escalas apropiadas.
- Buena resolución.

El análisis debe explicar los resultados observados, no limitarse a describir las gráficas.

No incluyas resultados que no hayan sido obtenidos.

## 11. Bitácora de inteligencia artificial

El documento exige expresamente una bitácora de IA.

Crea `report/bitacora_ia.md`.

Debe incluir:

1. Herramientas utilizadas.
2. Nombre y versión o modelo cuando se conozcan.
3. Propósito de cada herramienta.
4. Prompts completos enviados, en orden cronológico.
5. Respuestas o propuestas relevantes de la IA.
6. Cambios realizados sobre las propuestas.
7. Justificación de los cambios.
8. Errores o sugerencias incorrectas detectadas.
9. Cómo se corrigieron.
10. Explicación personal del funcionamiento del kernel CUDA.
11. Decisiones técnicas y justificaciones.
12. Procedimientos de verificación.

IMPORTANTE:

No inventes conversaciones, prompts ni respuestas que no hayan ocurrido.

Registra fielmente las interacciones reales de esta sesión. Cuando el historial de una interacción no esté disponible, marca ese apartado como pendiente de completar por el estudiante.

Puedes proponer explicaciones para que el estudiante las comprenda, pero no las presentes como explicaciones personales ya redactadas por él.

## 12. Documentación para aprender y defender el proyecto

Además del código y el informe, crea un documento:

`README.md`

Debe explicar:

- Qué problema resuelve el programa.
- Cómo funciona el método de Monte Carlo.
- Cómo ejecutar cada nivel.
- Cómo instalar las dependencias.
- Cómo reproducir los experimentos.
- Cómo generar las gráficas.
- Cómo ejecutar las pruebas.
- Cómo ejecutar el nivel GPU.
- Qué hacer si no hay GPU CUDA disponible.

También crea:

`docs/guia_sustentacion.md`

Este documento debe servir como material de estudio y contener explicaciones detalladas de:

### Fundamentos matemáticos
- Qué es una integral de Monte Carlo.
- Por qué se utiliza un estimador basado en promedios.
- Qué hace la función indicadora.
- Qué significa el error absoluto.
- Por qué aparece la convergencia 1/√N.

### Computación paralela
- Diferencias entre ejecución secuencial y paralela.
- Qué es Numba.
- Qué hace `njit`.
- Qué hace `parallel=True`.
- Qué hace `prange`.
- Qué es una condición de carrera.
- Qué es una reducción.
- Qué significa speedup.
- Qué es overhead.

### CUDA
- Qué es CUDA.
- Qué es un kernel.
- Qué son los bloques y los hilos.
- Cómo se calcula el índice global de un hilo.
- Cómo funciona `xoroshiro128p`.
- Por qué cada hilo necesita su estado aleatorio.
- Cómo se acumulan las contribuciones.
- Cómo funciona la reducción.
- Diferencias entre float32 y float64.
- Qué son las transferencias CPU-GPU.
- Cómo se miden correctamente los tiempos.

Incluye preguntas posibles de sustentación con sus respuestas razonadas.

No redactes respuestas excesivamente técnicas sin explicar primero los conceptos fundamentales.

## 13. Metodología de trabajo

Trabaja por etapas:

**Etapa 1:** inspección del entorno y planificación.

**Etapa 2:** implementación secuencial y validación.

**Etapa 3:** implementación paralela en CPU y benchmarks.

**Etapa 4:** implementación CUDA y validación.

**Etapa 5:** experimentos comparativos y generación de resultados.

**Etapa 6:** informe, bitácora y documentación.

**Etapa 7:** revisión integral contra los criterios de evaluación.

Al terminar cada etapa:

1. Explica qué desarrollaste.
2. Indica qué archivos creaste o modificaste.
3. Explica las decisiones técnicas.
4. Ejecuta las pruebas correspondientes.
5. Identifica problemas pendientes.
6. Indica qué conceptos debo comprender para poder explicar esa parte.

No avances silenciosamente sobre errores importantes.

Si encuentras un problema que requiere una decisión del usuario, detente y explica las alternativas.

## 14. Estándares de calidad

Aplica los siguientes criterios durante todo el desarrollo:

- Código limpio, modular y legible.
- Nombres descriptivos.
- Comentarios útiles, no redundantes.
- Funciones con responsabilidades claras.
- Evitar complejidad innecesaria.
- Manejo correcto de errores.
- Dependencias justificadas.
- Resultados reproducibles.
- Separación entre código, experimentos y documentación.
- Validación antes de aceptar una implementación.

No optimices prematuramente sacrificando claridad o corrección.

La prioridad es:

1. Corrección matemática.
2. Corrección algorítmica.
3. Validación.
4. Medición confiable.
5. Rendimiento.
6. Documentación.

## 15. Revisión final contra la evaluación

Antes de declarar terminado el proyecto, construye una lista de verificación que contemple todos los criterios de evaluación del PDF:

| Criterio | Peso |
|---|---:|
| Nivel 1: programa correcto y validado | 20 % |
| Nivel 1: gráfica de convergencia y análisis | 15 % |
| Nivel 2: versión paralela correcta | 15 % |
| Nivel 2: mediciones y análisis de speedup | 15 % |
| Informe y sustentación | 15 % |
| Nivel 3: kernel CUDA, comparación y bitácora | 20 % |

Comprueba cada requisito individualmente.

No marques como completado ningún criterio que no tenga evidencia verificable.

## INSTRUCCIÓN FINAL

Comienza exclusivamente con la inspección del PDF, del workspace y del entorno de ejecución.

Después, presenta un diagnóstico inicial que incluya:

- Requisitos identificados.
- Estado del proyecto.
- Características del hardware.
- Disponibilidad de CUDA.
- Dependencias necesarias.
- Estructura propuesta.
- Posibles limitaciones.

Una vez presentado el diagnóstico, comienza la implementación del Nivel 1.

Quiero que desarrolles el proyecto de manera autónoma, pero que me expliques las decisiones importantes y me permitas comprender el trabajo a medida que avanzamos.

**El resultado final debe ser un proyecto académico funcional, científicamente correcto, reproducible y que yo pueda explicar sin depender de la inteligencia artificial durante la sustentación.**
</USER_REQUEST>
<ADDITIONAL_METADATA>
The current local time is: 2026-11-16T22:23:38-05:00.
</ADDITIONAL_METADATA>
<USER_SETTINGS_CHANGE>
The user changed setting `Model Selection` from None to Gemini 3.8 Flash (High). No need to comment on this change if the user doesn't ask about it. If reporting what model you are, please use a human readable name instead of the exact string.
</USER_SETTINGS_CHANGE>
````

---

### Prompt 2: Auditoría Técnica y Corrección de la Bitácora

````text
# Auditoría técnica y corrección de la bitácora de IA — Proyecto Monte Carlo

Actúa como un revisor técnico especializado en computación científica, Python, Numba, paralelización en CPU, CUDA y documentación académica.

Tu tarea es auditar y corregir el archivo `bitacora_ia.md` del proyecto, utilizando como referencia el enunciado `Ev3_2026-2.pdf`, el código fuente, los notebooks, los resultados y los registros disponibles en el workspace.

## 1. Objetivo principal

Corregir los errores conceptuales, técnicos, matemáticos y documentales encontrados en la bitácora, sin cambiar innecesariamente su estructura ni eliminar información válida.

La bitácora debe cumplir todos los requisitos del profesor y describir fielmente lo que se hizo durante el desarrollo.

**Regla fundamental: no inventes información.**

No afirmes que una prueba fue ejecutada, que un kernel compiló, que una GPU fue utilizada o que un resultado fue validado si no existe evidencia en los archivos o registros disponibles.

Cuando no puedas verificar algo:
- Identifica exactamente qué información falta.
- No presentes una suposición como un hecho.
- Escribe una observación pendiente de verificación.
- Indica qué comando, prueba o ejecución permitiría comprobarlo.

## 2. Inspección previa obligatoria

Antes de modificar cualquier archivo:

1. Inspecciona la estructura completa del proyecto.
2. Lee `bitacora_ia.md` y `Ev3_2026-2.pdf`.
3. Identifica los scripts de los tres niveles.
4. Localiza el kernel CUDA y sus funciones auxiliares.
5. Revisa las pruebas automatizadas.
6. Revisa los notebooks, especialmente el notebook destinado a Google Colab.
7. Examina las tablas, gráficas, resultados y registros de ejecución.
8. Comprueba si existe evidencia de ejecución real en una GPU NVIDIA.

No modifiques el código fuente durante esta tarea, salvo que yo lo autorice posteriormente. El objetivo principal es auditar y corregir la documentación.

## 3. Correcciones técnicas obligatorias

### 3.1. Registro de prompts completos

En la sección 2 de la bitácora, el prompt inicial aparece resumido mediante puntos suspensivos y una descripción entre corchetes.

Esto incumple el requisito del profesor de incluir el texto completo de los prompts enviados.

Debes:

- Buscar los prompts originales en los registros, archivos o historial disponible dentro del workspace.
- Incorporar el texto íntegro de cada prompt recuperable.
- Ordenarlos cronológicamente.
- Registrar los prompts posteriores que hayan producido cambios importantes en el código o en las decisiones técnicas.
- No reconstruir un prompt de memoria y presentarlo como original.

Si no puedes recuperar el texto original, indica que el registro disponible es parcial y solicita el texto faltante. No inventes el contenido.

### 3.2. Justificación del uso de `float64`

La justificación actual menciona cancelación catastrófica al acumular millones de contribuciones en `float32`.

Esta explicación es incorrecta para el problema actual, porque la función integranda:

\[
f(x,y)=\rho(x,y)r^2\mathbf{1}[r\leq R]
\]

produce contribuciones no negativas. No existe la cancelación de términos positivos y negativos que caracteriza ese fenómeno.

Sustituye la justificación por una explicación técnicamente correcta:

- `float64` ofrece mayor precisión de representación y acumulación.
- Puede reducir los errores de redondeo.
- Permite mantener una precisión consistente entre las implementaciones.
- Su rendimiento puede ser menor en algunas GPU, por lo que también es una decisión que implica un costo computacional.

No afirmes que `float64` garantiza estabilidad numérica absoluta.

### 3.3. Verificación de CUDA y CUDASIM

La bitácora presenta como verificada la compilación del kernel en una GPU real, aunque el problema reportado ocurrió durante la ejecución de CUDASIM.

Debes diferenciar claramente:

- Ejecución en CUDASIM.
- Compilación del kernel CUDA.
- Ejecución en una GPU NVIDIA real.
- Validación numérica del resultado obtenido.

Un fallo en CUDASIM no demuestra que el kernel funcione en GPU real.

Comprueba si existe evidencia de una ejecución real en Google Colab o en otro equipo con GPU NVIDIA.

Si no existe:
- No afirmes que el kernel fue ejecutado ni validado en hardware real.
- Describe el problema de CUDASIM utilizando el mensaje de error real, si está disponible.
- Indica que el notebook fue preparado para validación posterior.
- Deja la ejecución real como pendiente.

Tampoco atribuyas el error exclusivamente a NumPy 2.x sin evidencia suficiente que confirme esa causa.

### 3.4. Descripción de hilos y bloques CUDA

La bitácora utiliza el ejemplo:

\[
256\text{ bloques}\times256\text{ hilos}=65\,536
\]

La multiplicación es correcta, pero no debes llamar a esos 65.536 hilos «hilos concurrentes».

Son los hilos lanzados en la cuadrícula. La GPU los programa y ejecuta de acuerdo con sus recursos, multiprocesadores de streaming y capacidad de residencia.

Utiliza la terminología correcta.

Además, revisa el código real para verificar:
- Cómo se calcula `tid`.
- Cómo se determina el número de bloques.
- Cómo se determina el número de hilos por bloque.
- Qué sucede con los hilos cuyo índice excede el número de muestras.
- Si se usa un bucle convencional o una distribución *strided*.

### 3.5. Distribución de muestras por hilo

La bitácora utiliza:

\[
M=\left\lceil\frac{N}{\text{total\_hilos}}\right\rceil
\]

Comprueba que esta fórmula coincide con la implementación real.

Si el kernel utiliza un índice como:

\[
i=tid\cdot M+j
\]

comprueba que exista una condición para evitar procesar índices \(i\geq N\).

Si utiliza otra estrategia, modifica la explicación para describir exactamente esa estrategia.

No expliques una distribución de trabajo que no corresponda al código implementado.

### 3.6. Generadores aleatorios `xoroshiro128p`

La bitácora indica que `create_xoroshiro128p_states` inicializa los generadores separados por \(2^{64}\) pasos.

Mantén la explicación, pero precisa que se trata de subsecuencias separadas según el mecanismo de Numba y que la independencia se establece bajo las condiciones de uso documentadas.

Comprueba:
- Si se utiliza un estado por hilo.
- Si el arreglo de estados tiene capacidad suficiente.
- Si cada hilo utiliza su propio índice `tid`.
- Si los estados se inicializan correctamente.
- Si las semillas y los offsets se utilizan de forma coherente.

No afirmes que las secuencias son absolutamente independientes sin explicar las condiciones correspondientes.

### 3.7. Justificación de 256 hilos por bloque

La bitácora dice que 256 hilos garantizan alta ocupación.

Esto es incorrecto como afirmación general.

Explica que 256 es múltiplo de 32, el tamaño de warp de NVIDIA, y permite organizar los hilos en warps completos.

Sin embargo, la ocupación real depende de:
- Registros por hilo.
- Memoria compartida.
- Recursos disponibles en cada SM.
- Límites de bloques y warps residentes.

Presenta 256 como una configuración elegida y razonable, no como una garantía universal de alta ocupación.

### 3.8. Reducción de resultados parciales

Revisa el código real de reducción CUDA.

La bitácora describe que cada hilo escribe en:

```python
d_parciales[tid] = acc
```

Comprueba que:
- Cada hilo escriba en una posición única.
- El arreglo tenga tamaño suficiente.
- No existan accesos fuera de rango.
- La transferencia al host ocurra correctamente.
- La suma final se realice sobre todos los resultados parciales válidos.

No afirmes que la reducción final tarda una fracción de milisegundo si no existen mediciones que lo demuestren.

### 3.9. Error estadístico de Monte Carlo

La bitácora presenta:

\[
3\frac{\sigma_f}{\sqrt N}
\]

como criterio de validación del error de la integral.

El estimador utilizado es:

\[
I_N=A\frac{1}{N}\sum_{i=1}^{N}f(x_i,y_i)
\]

Por tanto, si \(\sigma_f\) es la desviación estándar de las contribuciones individuales \(f\), el error estándar de la integral es:

\[
SE(I_N)=\frac{A\sigma_f}{\sqrt N}
\]

Y el criterio aproximado de tres desviaciones estándar es:

\[
\left|I_N-I_{\mathrm{exacto}}\right|
\leq
3\frac{A\sigma_f}{\sqrt N}
\]

Sin embargo, antes de cambiar la fórmula, inspecciona el código y determina qué desviación estándar se está calculando.

Si el código calcula la desviación estándar de las contribuciones ya multiplicadas por \(A\), la fórmula puede ser diferente.

Deja explícita la definición de la variable utilizada.

Aclara que el criterio de tres sigma es una aproximación basada en la distribución normal, no una garantía matemática universal del 99,7 %.

### 3.10. Reproducibilidad

La bitácora afirma que las semillas permiten reproducción exacta de todos los experimentos.

Corrige esta afirmación.

Distingue:
- Repetibilidad de una implementación con la misma semilla y generador.
- Reproducibilidad bajo las mismas condiciones.
- Diferencias numéricas entre CPU y GPU por generadores distintos y órdenes diferentes de acumulación.

No prometas igualdad bit a bit entre implementaciones si no ha sido comprobada.

### 3.11. Medición de tiempos CUDA

Inspecciona cómo se miden realmente los tiempos.

Diferencia:
1. Inicialización de la GPU y estados aleatorios.
2. Transferencias de memoria.
3. Ejecución del kernel.
4. Transferencia de resultados parciales al host.
5. Reducción final en CPU.

Si se utilizan eventos CUDA para medir únicamente el kernel, indícalo expresamente.

Si se presentan tiempos totales, especifica qué operaciones incluyen.

No afirmes precisión de microsegundos como garantía del método de medición sin comprobar la resolución y configuración utilizadas.

### 3.12. Procesamiento por chunks

La bitácora afirma que procesar \(10^7\) muestras en bloques de \(10^6\) evita la recolección de basura.

La justificación principal debe ser el control del consumo de memoria temporal al trabajar con arreglos vectorizados.

Comprueba que el tamaño del chunk indicado coincida con el código y no atribuyas mejoras de rendimiento que no se hayan medido.

### 3.13. Dependencia `MSVCP140.dll`

Revisa el apartado que describe la copia manual de `MSVCP140.dll` a la carpeta de Python.

No presentes esta intervención como una solución estándar recomendada.

Documenta:
- Qué error se produjo.
- Qué archivo se copió.
- De dónde se obtuvo.
- Qué efecto tuvo la intervención.
- Si se verificó posteriormente la carga de Numba.

Si no existe información suficiente, indica que la solución fue una intervención local cuyo procedimiento y origen deben documentarse mejor.

Menciona como alternativa recomendada la instalación o reparación del Microsoft Visual C++ Redistributable oficial, sin afirmar que fue lo que se hizo si no hay evidencia.

## 4. Verificaciones adicionales

Comprueba las siguientes afirmaciones directamente en el código y los resultados:

### CPU
- Que `@njit(parallel=True)` y `prange` se utilicen realmente.
- Que la acumulación corresponda a una reducción válida.
- Que la ejecución paralela se haya comprobado.
- Que el número de núcleos/hilos informado corresponda al hardware real.
- Que el calentamiento JIT ocurra antes de las mediciones.

### Nivel secuencial
- Que el muestreo sea uniforme en el cuadrado.
- Que se utilice correctamente la función indicadora.
- Que el área sea \(A=4R^2\).
- Que se compare contra:

\[
I_{\mathrm{exacto}}=\frac{9}{10}\pi\rho_0R^4
\]

- Que se calcule correctamente el error absoluto.
- Que las gráficas y valores correspondan a ejecuciones reales.

### GPU
- Que el kernel utilice `numba.cuda`.
- Que no se use `np.random` dentro del kernel.
- Que se use correctamente el generador aleatorio por hilo.
- Que se compruebe la validez de los índices.
- Que la reducción sea correcta.
- Que el resultado sea estadísticamente compatible con CPU y con el valor analítico.

### Benchmarks
- Que se excluya el tiempo de compilación JIT de CPU.
- Que se realicen varias repeticiones.
- Que se explique el uso de la mediana.
- Que el speedup utilice:

\[
S=\frac{T_{\mathrm{secuencial}}}{T_{\mathrm{paralelo}}}
\]

- Que las tablas y gráficas no contengan datos inventados.
- Que se diferencie el tiempo de kernel del tiempo total de GPU.

## 5. Cumplimiento de la consigna académica

Contrasta la bitácora con todos los requisitos del apartado «Uso de herramientas de inteligencia artificial» del PDF.

La versión corregida debe incluir:

1. Herramientas utilizadas, versiones y propósito.
2. Prompts completos y cronológicos.
3. Respuestas y propuestas relevantes de IA.
4. Cambios aceptados, rechazados o modificados, con sus razones.
5. Errores detectados y sus correcciones.
6. Explicación del kernel con palabras comprensibles para el estudiante.
7. Decisiones técnicas justificadas.
8. Verificación de resultados y pruebas realizadas.

No elimines información válida simplemente para hacer el documento más corto.

## 6. Formato de trabajo

Trabaja de la siguiente manera:

**Fase 1 — Auditoría**

Antes de editar, presenta una tabla con:
- Apartado.
- Afirmación actual.
- Problema detectado.
- Evidencia encontrada.
- Corrección necesaria.

**Fase 2 — Corrección**

Modifica únicamente `bitacora_ia.md`.

Conserva:
- La estructura general.
- Los títulos que sigan siendo adecuados.
- Las explicaciones correctas.
- Los datos que estén respaldados por evidencia.

Corrige los errores identificados y mejora la precisión técnica.

**Fase 3 — Verificación**

Vuelve a leer el archivo completo y comprueba:
- Que no existan contradicciones internas.
- Que las fórmulas sean matemáticamente correctas.
- Que las afirmaciones coincidan con el código.
- Que no se hayan inventado ejecuciones o resultados.
- Que todos los requisitos documentales del profesor estén cubiertos.

**Fase 4 — Informe de cambios**

Al terminar, presenta:
1. Lista de modificaciones realizadas.
2. Lista de afirmaciones que no pudieron verificarse.
3. Información que todavía debo proporcionar.
4. Pruebas que debo ejecutar en Google Colab.
5. Confirmación de que no modificaste otros archivos.

## 7. Restricciones

- No inventes prompts.
- No inventes resultados experimentales.
- No inventes tiempos de ejecución.
- No inventes especificaciones del hardware.
- No afirmes que una GPU real ejecutó el kernel si no existe evidencia.
- No modifiques los scripts del proyecto en esta tarea.
- No elimines errores históricos que deban documentarse.
- No conviertas hipótesis en hechos.
- No afirmes que el proyecto cumple completamente la consigna si existen requisitos pendientes.

El resultado debe ser una bitácora técnicamente rigurosa, honesta, coherente y defendible durante la sustentación académica.
</USER_REQUEST>
<ADDITIONAL_METADATA>
The current local time is: 2026-11-16T22:50:06-05:00.
</ADDITIONAL_METADATA>
````

---

### Prompt 3: Cierre del Proyecto con Resultados Reales de Google Colab

````text
Termina el proyecto, aquí inserto los archivos con respecto a cuda y la ejecusión en GPU (colab)
[Adjuntos: tabla_benchmarks_gpu.csv, comparativa_rendimiento_global.csv, speedup_comparativo_global.png]
````

---

## 3. Respuestas y Propuestas Relevantes de la IA

1. **Diagnóstico del Entorno Local:**
   - La IA inspeccionó el entorno e identificó que la CPU es un procesador AMD Ryzen (arquitectura Zen) con 4 núcleos físicos y 8 procesadores lógicos (hilos), acompañado de 21.95 GB de memoria RAM física bajo Windows 11.
   - Detectó que el adaptador gráfico del sistema corresponde a una GPU integrada **AMD Radeon(TM) RX Vega 11 Graphics**, por lo que la comprobación `numba.cuda.is_available()` retornó `False`.
   - Propuso no inventar resultados ni presentar mediciones simuladas como datos reales de hardware. Planteó ejecutar los benchmarks de los Niveles 1 y 2 en la CPU local, mientras que para el Nivel 3 propuso estructurar el módulo [src/monte_carlo_gpu.py](file:///c:/Users/XSayP/OneDrive/Desktop/Algoritmos3/src/monte_carlo_gpu.py) y preparar un entorno reproducible en Google Colab ([notebooks/monte_carlo_gpu_colab.ipynb](file:///c:/Users/XSayP/OneDrive/Desktop/Algoritmos3/notebooks/monte_carlo_gpu_colab.ipynb)).

2. **Estrategia de Vectorización y Control de Memoria (Nivel 1):**
   - Para el estimador secuencial en NumPy, propuso particionar el procesamiento en bloques de tamaño fijo (`chunk_size = 1_000_000`) cuando $N$ es grande ($10^7$ o $10^8$).
   - La justificación técnica radica en el **control estricto de la memoria temporal**: generar de una sola vez arreglos vectorizados de $N = 10^7$ o $10^8$ en NumPy requeriría múltiples gigabytes de memoria continua para coordenadas ($x, y$), distancias al cuadrado ($r^2$), máscaras booleanas y evaluaciones de densidad, lo que provocaría alta presión sobre la memoria RAM y desbordamiento de las memorias caché L2/L3. El particionamiento en lotes de un millón de elementos mantiene la huella de memoria residente en decenas de megabytes.

3. **Paralelización en CPU con Numba (Nivel 2):**
   - Propuso el uso de `@njit(parallel=True)` junto con la directiva `prange`.
   - Explicó que Numba reconoce la acumulación `suma_contribuciones += ...` dentro de `prange` como un patrón de **reducción paralela**. Esto garantiza que cada hilo del *pool* mantenga un acumulador privado en registros de la CPU, evitando condiciones de carrera sin necesidad de sincronizaciones o candados explícitos.

4. **Diseño del Kernel CUDA y Flujo de Reducción (Nivel 3):**
   - Planteó el uso de `numba.cuda.random` con el algoritmo `xoroshiro128p` y la función `xoroshiro128p_uniform_float64` para generar coordenadas aleatorias directamente en memoria del dispositivo.
   - Diseñó una estrategia de reducción en dos fases: acumulación privada en registros de cada hilo, almacenamiento de los subtotales en un arreglo global de dispositivo `d_parciales[tid]` y transferencia de este vector al host para la sumatoria final en CPU, eliminando la contención por operaciones atómicas en memoria global de la GPU.

---

## 4. Cambios Realizados sobre las Propuestas y Justificación

1. **Resolución y Documentación de la Dependencia DLL (`MSVCP140.dll`):**
   - *Situación / Error detectado:* Tras instalar Numba 0.68.1 en Python 3.12 (Windows 11), la importación del módulo falló con el siguiente error:
     ```text
     ImportError: DLL load failed while importing _typeconv: No se puede encontrar el módulo especificado.
     ```
   - *Diagnóstico técnico:* La extensión compilada `_typeconv.cp312-win_amd64.pyd` requería el runtime de C++ de Microsoft (`MSVCP140.dll`), ausente en el directorio de la aplicación de Python y en las rutas estándar del sistema.
   - *Acción realizada:* Se localizó una copia existente de `msvcp140.dll` en la ruta:
     `C:\Program Files\Microsoft OneDrive\26.204.1019.0007\msvcp140.dll`
     y se copió al directorio raíz del intérprete de Python:
     `C:\Users\XSayP\AppData\Local\Programs\Python\Python312\msvcp140.dll`
     Tras esta acción, `import numba` funcionó correctamente.
   - *Justificación y advertencia técnica:* Se documenta explícitamente que esta copia manual fue una **intervención puntual y ad-hoc** para habilitar la ejecución en el entorno local. No constituye la solución estándar recomendada. La práctica estándar recomendada en sistemas Windows consiste en **instalar o reparar el paquete oficial Microsoft Visual C++ 2015-2022 Redistributable (x64)** desde el instalador oficial de Microsoft.

2. **Aislamiento del Tiempo de Compilación JIT (*Warm-up*):**
   - *Justificación:* Numba compila las funciones a código máquina nativo (o PTX en CUDA) en la primera invocación (*Just-In-Time*). Si se mide la primera llamada, el tiempo registrado incluye el análisis de tipos, la optimización y la generación de código binario por LLVM, lo que falsea completamente el rendimiento, especialmente para tamaños pequeños como $N = 10^4$. Se implementó la función `calentar_numba_cpu()` para forzar la compilación previa antes de iniciar cualquier medición.

3. **Uso de la Mediana en vez de la Media para Benchmarks:**
   - *Justificación:* En sistemas operativos multiproceso de escritorio (como Windows 11), interrupciones del kernel, servicios en segundo plano y cambios dinámicos de frecuencia en la CPU introducen valores atípicos (*outliers*). La mediana es una medida de tendencia central más robusta que la media aritmética para caracterizar el tiempo de ejecución representativo.

4. **Distinción entre Mediciones Locales y Estado de GPU:**
   - *Justificación:* Al no disponer de hardware NVIDIA CUDA en el equipo local, se acordó explícitamente no reportar tiempos simulados como si fueran datos empíricos de GPU. Los resultados de CPU se registraron en su totalidad y el benchmark de GPU se dejó formalmente como pendiente de ejecución en Google Colab.

---

## 5. Errores y Sugerencias Incorrectas Detectadas y su Corrección

1. **Error durante la Prueba con el Simulador de CUDA (`CUDASIM`):**
   - *Error real observado:* Al probar la ejecución del generador aleatorio bajo el simulador de Numba CUDA (`NUMBA_ENABLE_CUDASIM=1`), el runtime de CPU JIT arrojó el siguiente error y advertencias:
     ```text
     NumbaWarning: Compilation is falling back to object mode WITHOUT looplifting enabled
     because Function "init_xoroshiro128p_states_cpu" failed type inference due to:
     Invalid use of type(CPUDispatcher(<function init_xoroshiro128p_state at ...>))
     with parameters (array(Record(s0[type=uint64;offset=0],s1[type=uint64;offset=8];16;True), 1d, C), Literal[int](0), uint64)
     ```
   - *Corrección conceptual y técnica:* Se identificó que este problema ocurrió en el simulador en CPU (`init_xoroshiro128p_states_cpu`), donde el despachador `@jit` de Numba en CPU falló al inferir el tipo estructurado `Record` del arreglo de estados. 
   - *Aclaración indispensable:* Un fallo en CUDASIM **no significa que el kernel falle en una GPU real**, ni una simulación exitosa garantiza que el kernel compile en hardware. En una GPU NVIDIA real, el código del kernel no pasa por el despachador de CPU, sino que es compilado directamente a PTX por NVVM y enlazado por el controlador CUDA. Se evitó atribuir el fallo exclusivamente a NumPy 2.x sin evidencia concluyente y se configuró el notebook de Google Colab para la validación definitiva sobre hardware NVIDIA.

2. **Corrección de la Justificación del Tipo de Dato `float64`:**
   - *Error detectado:* En una versión previa de la bitácora se afirmó que el uso de `float64` prevenía la «cancelación catastrófica» al acumular millones de contribuciones.
   - *Corrección técnica:* Dicha afirmación es matemáticamente incorrecta para este problema. La función integranda $f(x, y) = \rho(x, y) r^2 \mathbf{1}[r \le R]$ es estrictamente no negativa ($f(x, y) \ge 0$ para todo punto), por lo que no existe resta de cantidades positivas casi iguales (fenómeno definitorio de la cancelación catastrófica).
   - El motivo real para usar `float64` en lugar de `float32` es mitigar el **error de redondeo acumulativo** y el fenómeno de **absorción numérica**: cuando la suma parcial acumulada alcanza valores grandes y se le suman términos pequeños, la mantisa de 24 bits de `float32` ($\approx 7.2$ dígitos decimales) puede truncar o descartar los bits menos significativos del incremento. `float64` dispone de 53 bits de mantisa ($\approx 15.9$ dígitos decimales), ofreciendo mayor precisión de representación y acumulación, y manteniendo consistencia con las implementaciones de CPU.
   - Asimismo, se documentó que `float64` conlleva un **costo computacional apreciable en GPU**: en GPUs orientadas a consumidor o de precisión mixta (como GeForce o la serie T4), las unidades de doble precisión (FP64) operan a una fracción de la velocidad de FP32 (típicamente $1:32$ o $1:64$), por lo que la elección de `float64` implica un compromiso deliberado entre precisión y velocidad.

3. **Corrección de la Terminología de Hilos CUDA:**
   - *Error detectado:* Se describió previamente que $256 \text{ bloques} \times 256 \text{ hilos} = 65,536$ correspondían a «hilos concurrentes».
   - *Corrección:* $65,536$ es el número de **hilos totales lanzados en la cuadrícula (*Grid*)**. La concurrencia física real en el silicio está limitada por el número de Multiprocesadores de Streaming (SM) de la GPU y su capacidad de residencia de hilos y warps (por ejemplo, una GPU NVIDIA T4 cuenta con 40 SMs, permitiendo hasta 1024 hilos residentes por SM, sumando un máximo teórico de 40,960 hilos residentes simultáneos). Los bloques se despachan y ejecutan dinámicamente según la disponibilidad de recursos del hardware.

---

## 6. Explicación del Código del Kernel CUDA (Para la Sustentación)

> *Nota pedagógica:* Esta sección contiene la base conceptual que el estudiante debe estudiar y exponer personalmente ante el evaluador durante la sustentación oral.

### A. Generación de Números Pseudoaleatorios con `xoroshiro128p`
```python
u1 = xoroshiro128p_uniform_float64(rng_states, tid)
u2 = xoroshiro128p_uniform_float64(rng_states, tid)
```
- Cada hilo en la cuadrícula se identifica de forma unívoca mediante su índice lineal global `tid`.
- En la memoria de la GPU reside el arreglo `rng_states`, donde cada posición `rng_states[tid]` almacena el estado de 128 bits del algoritmo **xoroshiro128p+**.
- La función `create_xoroshiro128p_states(total_hilos, seed)` inicializa cada estado de modo que cada generador arranca en una **subsecuencia separada por $2^{64}$ pasos** respecto al generador adyacente (utilizando la función *jump* de xoroshiro128p).
- La independencia estadística entre hilos concurrentes está garantizada bajo la condición de uso documentada: ningún hilo individual debe consumir más de $2^{64}$ números pseudoaleatorios en su ciclo de vida (en nuestra simulación cada hilo genera del orden de $10^3$ a $10^5$ valores, muy por debajo de esa cota).
- La función retorna un valor flotante uniforme $u \in [0.0, 1.0)$, el cual se transforma afínmente al dominio del cuadrado $[-R, R]$ mediante $x = (2u - 1)R$.

### B. Distribución del Trabajo entre Hilos y Bloques
```python
tid = cuda.grid(1)  # tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
```
- La GPU organiza la cuadrícula en **bloques** (`blocks_per_grid`, por defecto 256) y cada bloque contiene **hilos** (`threads_per_block`, por defecto 256).
- El número total de hilos creados es:
  $$\text{total\_hilos} = \text{blocks\_per\_grid} \times \text{threads\_per\_block} = 65,536$$
- Cada hilo procesa un número entero fijo de muestras:
  $$M = \text{samples\_per\_thread} = \max\left(1, \left\lceil \frac{N_{\text{deseado}}}{\text{total\_hilos}} \right\rceil\right)$$
- El número efectivo de muestras evaluadas en la simulación es:
  $$N_{\text{efectivo}} = \text{total\_hilos} \times M \ge N_{\text{deseado}}$$
- Dentro del kernel, cada hilo verifica la condición de guarda `if tid < d_parciales.size:` y ejecuta un **bucle secuencial convencional** de $M$ iteraciones:
  ```python
  for _ in range(n_samples_per_thread):
      # Generación de coordenadas, evaluación de indicadora y acumulación
  ```
  La normalización final divide por $N_{\text{efectivo}}$, garantizando la validez estadística del estimador sin sesgo.

### C. Reducción y Acumulación de Resultados Parciales
```python
d_parciales[tid] = acc
```
- Si miles de hilos intentaran actualizar concurrentemente una única variable global en la memoria de la GPU, se generaría una severa contención en el controlador de memoria o se requerirían operaciones atómicas serializadas.
- En su lugar, se implementó una **reducción en dos fases**:
  1. **Fase 1 (Device / Hilos):** Cada hilo acumula localmente sus $M$ evaluaciones en un registro privado `acc` de doble precisión (`float64`). Al terminar sus iteraciones, escribe su subtotal en su celda exclusiva `d_parciales[tid]`. No hay colisiones de memoria ni carreras entre hilos.
  2. **Fase 2 (Host / CPU):** El vector `d_parciales` (de tamaño $65,536$ elementos) se transfiere de la GPU al host mediante `d_parciales.copy_to_host()`. La CPU realiza la suma reductora sobre todos los resultados parciales mediante `np.sum(h_parciales)`.

---

## 7. Decisiones Técnicas y Justificaciones

1. **Configuración de 256 Hilos por Bloque:**
   - Se seleccionó 256 porque es un múltiplo entero de 32 (el tamaño del *warp* de NVIDIA). Esto asegura que todos los warps lanzados estén completos, evitando hilos ociosos por división inexacta.
   - *Aclaración de ocupación:* 256 hilos por bloque es una elección estándar y razonable para equilibrar la granularidad del trabajo, pero **no garantiza por sí sola una alta ocupación del SM**. La ocupación real en hardware depende de la cantidad de registros asignados por hilo por el compilador (`ptxas`), el uso de memoria compartida y los límites arquitecturales de bloques y warps residentes por SM en la GPU donde se ejecute.

2. **Selección del Tipo de Dato (`float64`):**
   - Se eligió `float64` para mitigar la pérdida de precisión por error de redondeo acumulativo y absorción al sumar decenas de millones de términos positivos en el acumulador de cada hilo y en la reducción global.
   - Permite mantener paridad y coherencia numérica estricta con las implementaciones de referencia en CPU (NumPy y Python operan por defecto con flotantes de 64 bits).
   - Se documenta el compromiso de rendimiento: en GPUs con baja tasa de cómputo en FP64 (como GPUs de arquitectura Turing o Ampere orientadas a gráficos), la velocidad de cómputo es menor en comparación con `float32`.

3. **Muestras por Hilo ($M$):**
   - Al asignar $M$ muestras por hilo mediante un bucle interno, se reutiliza el mismo conjunto de estados aleatorios `rng_states` asignado inicialmente. Esto evita la necesidad de dimensionar un estado RNG por cada una de las $10^7$ o $10^8$ muestras, lo cual consumiría gigabytes innecesarios de memoria en la GPU.

4. **Desglose de Mediciones Temporales en GPU:**
   - En [src/monte_carlo_gpu.py](file:///c:/Users/XSayP/OneDrive/Desktop/Algoritmos3/src/monte_carlo_gpu.py) se definieron variables temporales separadas:
     - $T_{\text{rng}}$: Inicialización de estados de hilos con `create_xoroshiro128p_states` (medido con `time.perf_counter()` en CPU con sincronización).
     - $T_{\text{kernel}}$: Tiempo de ejecución física del kernel en la GPU, medido exclusivamente mediante **Eventos CUDA** (`cuda.event()`) con sincronización explícita (`end_event.synchronize()`). La función `elapsed_time()` reporta la duración medida por el reloj del silicio de la GPU en milisegundos.
     - $T_{\text{transfer}}$: Transferencia del vector de resultados desde la memoria global de la GPU hacia el host (`copy_to_host()`).
     - $T_{\text{total}}$: Duración integral del pipeline desde el inicio de la preparación hasta la reducción final.

---

## 8. Verificación, Reproducibilidad y Estado de Validación

1. **Pruebas de Límites Matemáticos y Geometría:**
   - Evaluadas y aprobadas en [tests/test_sequential.py](file:///c:/Users/XSayP/OneDrive/Desktop/Algoritmos3/tests/test_sequential.py):
     - Origen ($r=0$): Contribución $= 0.0$.
     - Borde del disco ($r=1$): $\rho = 1 \cdot (1 + 1) = 2.0$, contribución $= 2.0 \cdot 1^2 = 2.0$.
     - Fuera del disco ($r > 1$): Contribución $= 0.0$.
     - Fracción de puntos capturados dentro del disco: converge a $\pi/4 \approx 0.785398$ con tolerancia $< 1\%$.

2. **Pruebas de Concurrencia en CPU:**
   - Evaluadas y aprobadas en [tests/test_cpu.py](file:///c:/Users/XSayP/OneDrive/Desktop/Algoritmos3/tests/test_cpu.py): verificación de 8 hilos activos en Numba, consistencia numérica entre ejecución serial y paralela, y verificación de que la reducción no produce `NaN`, valores infinitos ni lecturas inconsistentes por condiciones de carrera.

3. **Criterio Estadístico de Monte Carlo (3-Sigma):**
   - El estimador de la integral es $\hat{I}_N = A \cdot \frac{1}{N} \sum_{i=1}^N f(x_i, y_i)$, donde $A = 4R^2 = 4.0$ es el área del cuadrado circunscrito y $f(x_i, y_i) = \rho(x_i, y_i) r_i^2 \mathbf{1}[r_i \le R]$.
   - Si $\sigma_f$ representa la desviación estándar muestral de las evaluaciones de $f$, el error estándar formal de la integral es:
     $$SE(\hat{I}_N) = \frac{A \cdot \sigma_f}{\sqrt{N}}$$
   - La prueba unitaria en `tests/test_sequential.py` verifica el criterio:
     $$|\hat{I}_N - I_{\text{exacto}}| \le 3 \cdot SE(\hat{I}_N) = 3 \frac{A \cdot \sigma_f}{\sqrt{N}}$$
   - *Aclaración:* El criterio de tres desviaciones estándar es una **aproximación asintótica** basada en la convergencia a una distribución normal por el Teorema del Límite Central (que asocia una probabilidad teórica del $\approx 99.7\%$ bajo normalidad perfecta), no una cota determinista absoluta.

4. **Reproducibilidad y Repetibilidad:**
   - **Repetibilidad idéntica (bit a bit):** Se garantiza únicamente dentro de la misma implementación (ej. NumPy con `seed` fijada o Numba serial en CPU) bajo el mismo entorno y versiones.
   - **Diferencias esperadas entre CPU y GPU:** Las implementaciones de CPU (NumPy / Numba) y GPU (`xoroshiro128p`) utilizan familias de algoritmos generadores pseudoaleatorios distintos; por tanto, no producen ni deben producir la misma secuencia de números pseudoaleatorios.
   - **Aritmética de punto flotante en paralelo:** En ejecuciones multi-hilo (CPU o GPU), el orden de asociación en las sumas flotantes puede diferir entre corridas debido a la planificación dinámica de hilos. Dado que la suma flotante no es asociativa $((a+b)+c \neq a+(b+c))$, pueden presentarse variaciones en los últimos dígitos menos significativos sin afectar la validez estadística del estimador.

5. **Estado de Validación del Nivel 3 (GPU):**
   - **ESTADO:** `COMPLETADO Y VALIDADO EN GOOGLE COLAB`.
   - *Evidencia empírica incorporada al workspace:*
     - Módulo de código fuente implementado y documentado: [src/monte_carlo_gpu.py](file:///c:/Users/XSayP/OneDrive/Desktop/Algoritmos3/src/monte_carlo_gpu.py).
     - Pruebas unitarias de importación y manejo de excepciones aprobadas: [tests/test_gpu.py](file:///c:/Users/XSayP/OneDrive/Desktop/Algoritmos3/tests/test_gpu.py).
     - Notebook interactivo ejecutado en Google Colab: [notebooks/monte_carlo_gpu_colab.ipynb](file:///c:/Users/XSayP/OneDrive/Desktop/Algoritmos3/notebooks/monte_carlo_gpu_colab.ipynb).
     - Resultados experimentales reales guardados en: [results/tables/tabla_benchmarks_gpu.csv](file:///c:/Users/XSayP/OneDrive/Desktop/Algoritmos3/results/tables/tabla_benchmarks_gpu.csv) y [results/tables/comparativa_rendimiento_global.csv](file:///c:/Users/XSayP/OneDrive/Desktop/Algoritmos3/results/tables/comparativa_rendimiento_global.csv).
     - Gráfica comparativa global generada en Colab y guardada en: [results/figures/speedup_comparativo_global.png](file:///c:/Users/XSayP/OneDrive/Desktop/Algoritmos3/results/figures/speedup_comparativo_global.png).
   - *Resumen de hallazgos observados en GPU:*
     - Para $N = 10^8$, la GPU obtuvo $I_{\text{GPU}} = 2.827450$ (error absoluto $= 1.62 \times 10^{-5}$), demostrando convergencia numérica y consistencia estadística con la CPU y con el valor analítico exacto $0.9\pi \approx 2.827433$.
     - El tiempo de kernel para $N = 10^8$ fue $65.81\text{ ms}$ ($78.99\text{ ms}$ totales), alcanzando un *speedup* de **$54.86\times$** (kernel vs CPU serial), **$45.71\times$** (total end-to-end vs CPU serial) y **$10.67\times$** (kernel vs CPU paralelo de 8 hilos).
     - Se corroboró experimentalmente que para tamaños pequeños ($N \le 10^5$), el tiempo de inicialización de los 65,536 estados `xoroshiro128p` ($T_{\text{rng}} \approx 13\text{ ms}$) domina el tiempo total de GPU, generando *speedups* totales menores que 1 ($0.03\times$ y $0.29\times$).
