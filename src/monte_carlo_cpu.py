"""
Implementaciones en CPU utilizando Numba:
1. Versión secuencial compilada JIT (@njit(parallel=False))
2. Versión paralela multi-hilo (@njit(parallel=True), prange y reducción automática)

Nivel 2 - Algoritmos y Arquitecturas Computacionales de Alto Rendimiento (UIS).
"""

import math
import numpy as np
from numba import njit, prange
from typing import Dict, Any, Optional

from src.config import R, RHO_0, AREA_CUADRADO, I_EXACTO, DEFAULT_SEED


@njit(parallel=False, fastmath=False)
def _kernel_monte_carlo_serial(n_muestras: int, r_max: float, rho_0: float) -> float:
    """
    Kernel computacional serial en Numba.
    Ejecuta un bucle secuencial clásico en un único núcleo de CPU.
    """
    r_max_sq = r_max * r_max
    suma_contribuciones = 0.0

    for _ in range(n_muestras):
        # Muestreo uniforme en [-R, R]
        x = np.random.uniform(-r_max, r_max)
        y = np.random.uniform(-r_max, r_max)

        r2 = x * x + y * y
        # Función indicadora 1[r <= R]
        if r2 <= r_max_sq:
            r = math.sqrt(r2)
            rho = rho_0 * (1.0 + r / r_max)
            suma_contribuciones += rho * r2

    return suma_contribuciones


@njit(parallel=True, fastmath=False)
def _kernel_monte_carlo_parallel(n_muestras: int, r_max: float, rho_0: float) -> float:
    """
    Kernel computacional paralelo en Numba utilizando multi-hilos (prange).

    Aspectos clave de arquitectura:
    1. prange divide el rango [0, n_muestras) en bloques contiguos entre los hilos del pool.
    2. Numba identifica 'suma_contribuciones' como una variable de reducción.
    3. Cada hilo mantiene un acumulador privado local (evitando condiciones de carrera).
    4. Al finalizar el bucle, Numba realiza la suma de los acumuladores privados (reducción).
    5. Cada hilo dispone de su propio estado del generador pseudoaleatorio interno.
    """
    r_max_sq = r_max * r_max
    suma_contribuciones = 0.0

    for _ in prange(n_muestras):
        x = np.random.uniform(-r_max, r_max)
        y = np.random.uniform(-r_max, r_max)

        r2 = x * x + y * y
        if r2 <= r_max_sq:
            r = math.sqrt(r2)
            rho = rho_0 * (1.0 + r / r_max)
            suma_contribuciones += rho * r2

    return suma_contribuciones


def calentar_numba_cpu():
    """
    Realiza una ejecución de calentamiento (warm-up) con un tamaño pequeño
    para disparar la compilación JIT (Just-In-Time) de ambos kernels.
    Evita que el tiempo de compilación LLVM contamine las mediciones de rendimiento.
    """
    _kernel_monte_carlo_serial(100, R, RHO_0)
    _kernel_monte_carlo_parallel(100, R, RHO_0)


def monte_carlo_cpu_serial(
    n_muestras: int,
    r_max: float = R,
    rho_0: float = RHO_0,
    seed: Optional[int] = DEFAULT_SEED,
) -> Dict[str, Any]:
    """
    Envoltura de alto nivel para la ejecución serial en CPU con Numba.
    """
    if seed is not None:
        np.random.seed(seed)

    area = 4.0 * (r_max ** 2)
    suma = _kernel_monte_carlo_serial(n_muestras, r_max, rho_0)
    i_estimado = area * (suma / n_muestras)
    error_abs = abs(i_estimado - I_EXACTO)

    return {
        "n_muestras": n_muestras,
        "i_estimado": i_estimado,
        "i_exacto": I_EXACTO,
        "error_absoluto": error_abs,
        "error_relativo_pct": (error_abs / I_EXACTO) * 100.0,
    }


def monte_carlo_cpu_parallel(
    n_muestras: int,
    r_max: float = R,
    rho_0: float = RHO_0,
    seed: Optional[int] = DEFAULT_SEED,
) -> Dict[str, Any]:
    """
    Envoltura de alto nivel para la ejecución paralela multi-hilo en CPU con Numba.
    """
    if seed is not None:
        np.random.seed(seed)

    area = 4.0 * (r_max ** 2)
    suma = _kernel_monte_carlo_parallel(n_muestras, r_max, rho_0)
    i_estimado = area * (suma / n_muestras)
    error_abs = abs(i_estimado - I_EXACTO)

    return {
        "n_muestras": n_muestras,
        "i_estimado": i_estimado,
        "i_exacto": I_EXACTO,
        "error_absoluto": error_abs,
        "error_relativo_pct": (error_abs / I_EXACTO) * 100.0,
    }


if __name__ == "__main__":
    import time
    print("Iniciando calentamiento JIT...")
    calentar_numba_cpu()
    print("Calentamiento finalizado.")

    n = 1_000_000
    t0 = time.perf_counter()
    res_s = monte_carlo_cpu_serial(n)
    t_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    res_p = monte_carlo_cpu_parallel(n)
    t_p = time.perf_counter() - t0

    print(f"\nResultados N={n:,}:")
    print(f"Serial:   I={res_s['i_estimado']:.6f}, Error={res_s['error_absoluto']:.6f}, Tiempo={t_s*1000:.2f} ms")
    print(f"Paralelo: I={res_p['i_estimado']:.6f}, Error={res_p['error_absoluto']:.6f}, Tiempo={t_p*1000:.2f} ms")
    print(f"Speedup:  {t_s/t_p:.2f}x")
