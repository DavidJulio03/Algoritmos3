# Informe Académico: Integración de Monte Carlo y Computación de Alto Rendimiento
**Cálculo del Momento de Inercia de un Disco con Densidad Variable**

- **Institución:** Universidad Industrial de Santander (UIS)
- **Escuela:** Ingeniería de Sistemas e Informática
- **Asignatura:** Algoritmos y Arquitecturas Computacionales de Alto Rendimiento (2026-2)
- **Autor:** Estudiante UIS
- **Entorno Experimental:** AMD Ryzen (Zen Architecture, 4 núcleos físicos / 8 hilos lógicos), 22 GB RAM, Windows 11, Python 3.12.10, Numba 0.68.1, NumPy 2.5.5.

---

## 1. Introducción y Objetivo

El cálculo del momento de inercia en cuerpos con densidad superficial no homogénea es fundamental en mecánica clásica y física computacional. Cuando la geometría o la función de distribución de masa dificultan la resolución analítica, los métodos numéricos estocásticos (Monte Carlo) constituyen una herramienta de aproximación robusta.

El objetivo de esta evaluación consiste en implementar, validar, paralelizar y optimizar el cálculo del momento de inercia de un disco delgado de radio $R$ mediante integración de Monte Carlo a través de tres niveles de abstracción y arquitectura:
1. **Nivel 1:** Implementación secuencial en Python/NumPy validada rigurosamente contra la solución analítica exacta.
2. **Nivel 2:** Paralelización en CPU multinúcleo con compilación JIT utilizando Numba (`@njit(parallel=True)` y `prange`).
3. **Nivel 3:** Paralelización masiva en GPU utilizando Numba CUDA y generadores pseudoaleatorios `xoroshiro128p` independientes por hilo.

---

## 2. Descripción del Problema Físico y Método de Monte Carlo

Se considera un disco delgado de radio $R=1.0$ centrado en el origen, cuya densidad superficial de masa crece linealmente con la distancia radial $r = \sqrt{x^2+y^2}$:

$$\rho(x, y) = \rho_0 \left(1 + \frac{r}{R}\right), \quad \text{con } \rho_0 = 1.0$$

El momento de inercia respecto al eje $z$ perpendicular al disco viene dado por:

$$I = \iint_D \rho(x, y) \, r^2 \, dA$$

La solución analítica exacta se obtiene en coordenadas polares ($dA = r \, dr \, d\theta$):

$$I_{\text{exacto}} = \int_0^{2\pi} d\theta \int_0^R \rho_0 \left(1 + \frac{r}{R}\right) r^3 \, dr = 2\pi \rho_0 \left[ \frac{R^4}{4} + \frac{R^5}{5R} \right] = \frac{9}{10}\pi \rho_0 R^4$$

Con los parámetros dados $\rho_0 = 1$ y $R = 1$:

$$I_{\text{exacto}} = 0.9\pi \approx 2.82743338823$$

### Estimador de Monte Carlo en Coordenadas Cartesianas
Dado que un disco no es un rectángulo, se circunscribe en un cuadrado $[-R, R] \times [-R, R]$ de área $A = 4R^2 = 4.0$. Se muestrean uniformemente $N$ puntos aleatorios $(x_i, y_i) \sim \mathcal{U}(-R, R)$ y se aplica la función indicadora $\mathbf{1}[r_i \le R]$:

$$I \approx A \cdot \frac{1}{N} \sum_{i=1}^N \rho(x_i, y_i) \, r_i^2 \cdot \mathbf{1}[r_i \le R]$$

---

## 3. Implementación Secuencial y Validación (Nivel 1)

Se implementó el estimador en `src/monte_carlo_sequential.py`, empleando procesamiento por lotes para evitar un consumo desmedido de memoria RAM ante $N$ elevados. Se validó la función indicadora en puntos analíticos frontera ($r=0, r=1, r>1$) y la proporción de puntos capturados dentro del disco, convergiendo a $\frac{\pi R^2}{4R^2} = \frac{\pi}{4} \approx 78.54\%$.

Para el análisis de convergencia se evaluaron $N = 10^3, 10^4, 10^5, 10^6, 10^7$ realizando 10 repeticiones independientes con semillas pseudoaleatorias controladas.

### Tabla 1: Resumen de convergencia experimental (Nivel 1)

| $N$ | $\bar{I}_{\text{estimado}}$ | Desv. Est. $\hat{\sigma}_I$ | Error Abs. Mediana | Error Abs. Media | Error Relativo (%) | Error Teórico $\frac{A\sigma_f}{\sqrt{N}}$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| $10^3$ | 2.841960 | 0.092704 | 0.041763 | 0.068051 | 2.41 % | 0.081354 |
| $10^4$ | 2.821521 | 0.022135 | 0.019257 | 0.018306 | 0.65 % | 0.025522 |
| $10^5$ | 2.825777 | 0.008832 | 0.008029 | 0.007385 | 0.26 % | 0.008078 |
| $10^6$ | 2.826583 | 0.002542 | 0.001342 | 0.002050 | 0.07 % | 0.002552 |
| $10^7$ | 2.827340 | 0.000684 | 0.000543 | 0.000590 | 0.02 % | 0.000807 |

### Gráfica de Convergencia del Error (Escala Log-Log)
En la figura generada ([convergencia_error_nivel1.png](file:///c:/Users/XSayP/OneDrive/Desktop/Algoritmos3/results/figures/convergencia_error_nivel1.png)), el error empírico sigue con exactitud la recta de referencia con pendiente $-1/2$ ($\propto N^{-1/2}$). Esto confirma el **Teorema del Límite Central**: la desviación estándar del promedio muestral decae a razón de $\frac{1}{\sqrt{N}}$, requiriendo multiplicar por 100 el número de muestras para ganar un orden de magnitud (factor de 10) en precisión decimal.

---

## 4. Paralelización en CPU con Numba (Nivel 2)

Se implementó el kernel paralelo en `src/monte_carlo_cpu.py` usando `@njit(parallel=True)` y `prange`.
- **Prevención de condiciones de carrera y reducción:** Numba compila la variable acumuladora `suma_contribuciones += ...` dentro de `prange` como una reducción paralela. Cada hilo acumula de forma privada en registros locales de la CPU; al terminar la partición del bucle, el runtime realiza una suma reductora entre los hilos del pool.
- **Protocolo de medición:** Se ejecutó un calentamiento JIT (*warm-up*) previo para aislar la compilación LLVM. Las mediciones de tiempo se efectuaron con `time.perf_counter()` sobre 5 repeticiones independientes por configuración, extrayendo la **mediana** para eliminar fluctuaciones del sistema operativo.

### Tabla 2: Tiempos de ejecución y Speedup en CPU (8 hilos hardware)

| $N$ | $T_{\text{NumPy\_Sec}}$ (s) | $T_{\text{Numba\_Serial}}$ (s) | $T_{\text{Numba\_Paralelo}}$ (s) | Speedup CPU ($T_{\text{ser}} / T_{\text{par}}$) | Speedup Global ($T_{\text{NumPy}} / T_{\text{par}}$) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| $10^4$ | 0.000308 | 0.000465 | 0.000139 | **3.35x** | 2.22x |
| $10^5$ | 0.004009 | 0.004079 | 0.000684 | **5.96x** | 5.86x |
| $10^6$ | 0.050116 | 0.035185 | 0.007451 | **4.72x** | 6.73x |
| $10^7$ | 0.596169 | 0.361734 | 0.067689 | **5.34x** | 8.81x |
| $10^8$ | 5.623193 | 3.610525 | 0.702136 | **5.14x** | 8.01x |

### Gráfica de Aceleración (Speedup)
En la gráfica ([speedup_cpu.png](file:///c:/Users/XSayP/OneDrive/Desktop/Algoritmos3/results/figures/speedup_cpu.png)), el speedup en CPU se estabiliza entre **$5.1\times$ y $5.9\times$**. En una máquina de 4 núcleos físicos y 8 hilos con Simultaneous Multithreading (SMT), el límite teórico real no alcanza el $8.0\times$ perfecto debido a que los hilos lógicos emparejados comparten las unidades de coma flotante (ALU/FPU) y la jerarquía de memoria caché L1/L2. Para tamaños reducidos ($N \le 10^4$), la sobrecarga (*overhead*) de distribución de hilos y sincronización en barrera de Numba disminuye la eficiencia relativa respecto al cómputo puro.

---

## 5. Implementación y Resultados en GPU con Numba CUDA (Nivel 3)

La solución en GPU (`src/monte_carlo_gpu.py` y `notebooks/monte_carlo_gpu_colab.ipynb`) se ejecutó en hardware NVIDIA en Google Colab:
1. **Generación Pseudoaleatoria:** En GPU no es viable `np.random`. Se utilizó `numba.cuda.random` con el generador `xoroshiro128p`. Cada hilo lanzado en la cuadrícula accede a su propio estado estocástico mediante `create_xoroshiro128p_states` (subsecuencias separadas $2^{64}$ pasos).
2. **Jerarquía de Ejecución y Mapeo:** Se configuró una cuadrícula 1D de $256 \text{ bloques} \times 256 \text{ hilos} = 65,536$ hilos totales (múltiplo del warp de 32 hilos), procesando $M = \lceil N / 65,536 \rceil$ muestras por hilo en doble precisión (`float64`) para mitigar errores de redondeo acumulativo.
3. **Estrategia de Reducción en Dos Fases:** Cada hilo acumula su subtotal en un registro privado y lo escribe en `d_parciales[tid]` sin contención atómica. Posteriormente, el arreglo se transfiere al host (`copy_to_host()`) para la suma final.

### Tabla 3: Resultados experimentales en GPU (Google Colab) y comparativa de Speedup

| $N$ | $I_{\text{GPU}}$ | Error Abs. GPU | $T_{\text{rng}}$ (ms) | $T_{\text{kernel}}$ (ms) | $T_{\text{trans}}$ (ms) | $T_{\text{total}}$ (ms) | Speedup Kernel vs CPU Ser | Speedup Total vs CPU Ser | Speedup Kernel vs CPU Par |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| $10^4$ | 2.825289 | 0.002144 | 13.22 | 0.21 | 0.19 | 13.81 | 2.24x | 0.03x | 0.67x |
| $10^5$ | 2.826078 | 0.001355 | 13.64 | 0.23 | 0.18 | 14.29 | 18.07x | 0.29x | 3.03x |
| $10^6$ | 2.829979 | 0.002546 | 13.08 | 0.90 | 0.18 | 14.54 | 39.01x | 2.42x | 8.26x |
| $10^7$ | 2.827260 | 0.000173 | 12.88 | 7.49 | 0.21 | 20.94 | 48.28x | 17.28x | 9.03x |
| $10^8$ | 2.827450 | 0.000016 | 12.73 | 65.81 | 0.22 | 78.99 | **54.86x** | **45.71x** | **10.67x** |

### Análisis Comparativo Global y Gráfica de Speedup
En la figura global ([speedup_comparativo_global.png](file:///c:/Users/XSayP/OneDrive/Desktop/Algoritmos3/results/figures/speedup_comparativo_global.png)), se observan dos regímenes bien diferenciados:
- **Dominio del *overhead* ($N \le 10^5$):** La inicialización de los 65,536 estados `xoroshiro128p` toma un costo casi constante de $T_{\text{rng}} \approx 12.7 - 13.6\text{ ms}$. Para $N = 10^4$ y $10^5$, este tiempo de preparación supera ampliamente la duración del cálculo en CPU, provocando que el *speedup* total sea menor que 1 ($0.03\times$ y $0.29\times$).
- **Escalabilidad masiva ($N \ge 10^6$):** A medida que aumenta la carga computacional, la sobrecarga de inicialización y transferencia ($T_{\text{trans}} \approx 0.2\text{ ms}$) se amortiza. Para $N = 10^8$, el kernel GPU resuelve los cien millones de muestras en **$65.81\text{ ms}$** ($78.99\text{ ms}$ totales), logrando una aceleración de **$54.86\times$** frente a CPU serial (**$45.71\times$** considerando todo el pipeline end-to-end) y superando en **$10.67\times$** a la versión paralela en CPU de 8 hilos, con un error absoluto de apenas $1.62 \times 10^{-5}$.

---

## 6. Conclusiones

1. **Validez del Estimador:** La convergencia de Monte Carlo se validó en todas las arquitecturas (error relativo $< 0.02\%$ para $N=10^7$ en CPU y $0.00057\%$ para $N=10^8$ en GPU), confirmando la tasa teórica $\mathcal{O}(N^{-1/2})$.
2. **Impacto de la Paralelización en CPU:** Numba JIT con `prange` alcanzó una aceleración sostenida de hasta **$5.96\times$** en 8 hilos lógicos, saturando eficientemente las unidades de punto flotante del procesador.
3. **Superioridad y Amortización en GPU:** El kernel CUDA con `xoroshiro128p` y reducción en dos fases demostró una aceleración de **$54.86\times$** (kernel) y **$45.71\times$** (pipeline completo) en $N=10^8$, evidenciando cómo el paralelismo masivo requiere un volumen de carga suficiente para amortizar la inicialización de estados en el dispositivo.

---
*Fin del Informe Académico.*
