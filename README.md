# Cálculo del Momento de Inercia mediante Integración de Monte Carlo y Computación de Alto Rendimiento

**Universidad Industrial de Santander (UIS)**  
**Escuela de Física**  
**Asignatura:** Algoritmos y Arquitecturas Computacionales de Alto Rendimiento (2026-2)  
**Evaluación 3:** Integración de Monte Carlo en CPU y GPU  

---

## 1. Descripción del Problema

El proyecto calcula numéricamente el momento de inercia $I$ de un disco delgado de radio $R=1.0$ centrado en el origen con densidad superficial radial variable:

$$\rho(x, y) = \rho_0 \left(1 + \frac{r}{R}\right), \quad r = \sqrt{x^2+y^2}, \quad \rho_0 = 1.0$$

El momento de inercia respecto al eje perpendicular que pasa por su centro es:

$$I = \iint_D \rho(x, y) \, r^2 \, dA$$

### Solución Analítica de Referencia
$$I_{\text{exacto}} = \frac{9}{10}\pi \rho_0 R^4 = 0.9\pi \approx 2.82743338823$$

### Estimador de Monte Carlo
El disco se inscribe dentro del cuadrado $[-R, R] \times [-R, R]$ de área $A = 4R^2 = 4.0$. Se generan $N$ puntos uniformes $(x_i, y_i)$ y se calcula:

$$I \approx A \cdot \frac{1}{N} \sum_{i=1}^N \rho(x_i, y_i) \, r_i^2 \cdot \mathbf{1}[r_i \le R]$$

---

## 2. Estructura del Repositorio

```text
Algoritmos3/
│
├── README.md                      # Guía general de uso y reproducción
├── requirements.txt               # Dependencias del proyecto
├── .gitignore                     # Filtros de control de versiones
│
├── src/                           # Código fuente modular
│   ├── __init__.py
│   ├── config.py                  # Parámetros físicos, matemáticos y rutas
│   ├── monte_carlo_sequential.py  # Nivel 1: Implementación secuencial
│   ├── monte_carlo_cpu.py         # Nivel 2: Paralelización CPU Numba (prange)
│   ├── monte_carlo_gpu.py         # Nivel 3: Kernel GPU con Numba CUDA
│   ├── benchmarks.py              # Suite de medición y speedup CPU
│   ├── validation.py              # Experimentos de convergencia Nivel 1
│   └── generate_notebook.py       # Generador del notebook para Colab
│
├── results/                       # Salidas experimentales reproducibles
│   ├── raw/                       # Mediciones crudas en CSV
│   ├── tables/                    # Tablas procesadas (resúmenes y speedups)
│   └── figures/                   # Gráficas generadas en alta resolución
│
├── notebooks/                     # Entornos interactivos
│   └── monte_carlo_gpu_colab.ipynb# Notebook listo para Google Colab con GPU
│
├── report/                        # Entregables académicos
│   ├── informe.md                 # Informe corto de evaluación (máx 3 páginas)
│   └── bitacora_ia.md             # Bitácora obligatoria del uso de IA
│
└── tests/                         # Pruebas automatizadas (pytest)
    ├── test_sequential.py
    ├── test_cpu.py
    └── test_gpu.py
```

---

## 3. Instalación de Dependencias

Se requiere Python 3.10 o superior (validado en Python 3.12.10).

Instalar los paquetes necesarios:
```bash
pip install -r requirements.txt
```

---

## 4. Guía de Ejecución Rápida

### A. Nivel 1: Validación Secuencial y Gráfica de Convergencia
Ejecuta los experimentos para $N = 10^3, 10^4, 10^5, 10^6, 10^7$ (10 repeticiones por cada $N$), genera la tabla en CSV y la gráfica en escala log-log:
```bash
python -m src.validation
```
- Archivo generado: `results/figures/convergencia_error_nivel1.png`
- Tabla generada: `results/tables/tabla_convergencia_nivel1.csv`

### B. Nivel 2: Benchmarks de CPU y Gráficas de Speedup
Ejecuta la versión serial y la versión paralela multi-hilo (`prange`) con calentamiento previo JIT para $N = 10^4, 10^5, 10^6, 10^7, 10^8$:
```bash
python -m src.benchmarks
```
- Gráficas generadas: `results/figures/speedup_cpu.png` y `results/figures/tiempos_ejecucion_cpu.png`
- Tabla generada: `results/tables/tabla_speedup_cpu.csv`

### C. Nivel 3: Ejecución en GPU (Numba CUDA)
Para comprobar el subsistema CUDA localmente:
```bash
python -m src.monte_carlo_gpu
```

#### ¿Qué hacer si no hay GPU NVIDIA localmente?
Si tu máquina local cuenta con gráficos integrados (ej. AMD Radeon o Intel Iris Xe) o no dispone de una GPU NVIDIA con núcleos CUDA:
1. Abre [Google Colab](https://colab.research.google.com/).
2. Sube el notebook ubicado en: `notebooks/monte_carlo_gpu_colab.ipynb`.
3. Ve al menú `Entorno de ejecución` -> `Cambiar tipo de entorno de ejecución` y selecciona **T4 GPU**.
4. Ejecuta todas las celdas del notebook para obtener los resultados y gráficas con una GPU NVIDIA real.

---

## 5. Ejecución de la Suite de Pruebas

Para validar la corrección matemática, geométrica y de concurrencia:
```bash
python -m pytest tests/ -v
```
Se ejecutan 14 pruebas que validan:
- Constantes analíticas y función indicadora.
- Reproducibilidad con semillas estocásticas fijas.
- Ausencia de condiciones de carrera en la reducción con `prange`.
- Manejo correcto de excepciones en caso de ausencia de GPU.