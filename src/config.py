"""
Configuración global y constantes del problema físico y computacional.
"""

import math
from pathlib import Path

# ==============================================================================
# Parámetros físicos y matemáticos del problema
# ==============================================================================
R: float = 1.0          # Radio del disco delgado
RHO_0: float = 1.0      # Densidad superficial de referencia
AREA_CUADRADO: float = 4.0 * (R ** 2)  # Área del dominio de muestreo [-R, R] x [-R, R]

# Solución analítica exacta de referencia: I_exacto = (9/10) * pi * rho_0 * R^4
I_EXACTO: float = 0.9 * math.pi * RHO_0 * (R ** 4)

# ==============================================================================
# Tamaños de muestra para experimentos
# ==============================================================================
# Nivel 1 (Secuencial y convergencia de error)
N_VALORES_NIVEL_1 = [10**3, 10**4, 10**5, 10**6, 10**7]

# Nivel 2 y 3 (Benchmarks CPU Paralelo y GPU)
N_VALORES_BENCHMARKS = [10**4, 10**5, 10**6, 10**7, 10**8]

# ==============================================================================
# Configuración experimental y de reproducibilidad
# ==============================================================================
DEFAULT_SEED: int = 42
BENCHMARK_REPETICIONES: int = 5  # Número de corridas para calcular la mediana

# Rutas del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
RAW_RESULTS_DIR = RESULTS_DIR / "raw"
PROCESSED_RESULTS_DIR = RESULTS_DIR / "processed"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORTS_DIR = BASE_DIR / "report"
DOCS_DIR = BASE_DIR / "docs"

# ==============================================================================
# GPU CUDA Configuración predeterminada (para Nivel 3)
# ==============================================================================
CUDA_THREADS_PER_BLOCK: int = 256
