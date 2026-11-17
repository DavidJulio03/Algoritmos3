"""
Implementación secuencial del estimador de Monte Carlo para el cálculo
del momento de inercia de un disco delgado con densidad superficial variable.

Nivel 1 - Algoritmos y Arquitecturas Computacionales de Alto Rendimiento (UIS).
"""

import math
import numpy as np
from typing import Tuple, Dict, Any, Optional
from src.config import R, RHO_0, AREA_CUADRADO, I_EXACTO, DEFAULT_SEED


def calcular_contribucion_punto(
    x: float, y: float, r_max: float = R, rho_0: float = RHO_0
) -> float:
    """
    Evalúa la función integrando f(x, y) = rho(x, y) * r^2 * 1[r <= R]
    para un único punto en el plano cartesiano.

    Parámetros:
    -----------
    x, y : float
        Coordenadas del punto muestreado uniformemente en [-R, R] x [-R, R].
    r_max : float
        Radio del disco (R).
    rho_0 : float
        Densidad de referencia en el centro.

    Retorna:
    --------
    float : Contribución del punto al momento de inercia (0 si está fuera del disco).
    """
    r_cuadrado = x * x + y * y
    if r_cuadrado <= r_max * r_max:
        r = math.sqrt(r_cuadrado)
        rho = rho_0 * (1.0 + r / r_max)
        return rho * r_cuadrado
    return 0.0


def monte_carlo_secuencial_puro(
    n_muestras: int,
    r_max: float = R,
    rho_0: float = RHO_0,
    seed: Optional[int] = DEFAULT_SEED,
) -> Tuple[float, float]:
    """
    Implementación secuencial básica con bucle for explícito en Python puro.
    Ideal para comprensión conceptual y verificación paso a paso.

    Retorna:
    --------
    Tuple[float, float] : (I_estimado, error_absoluto)
    """
    if seed is not None:
        np.random.seed(seed)

    suma = 0.0
    area = 4.0 * (r_max ** 2)

    for _ in range(n_muestras):
        x = np.random.uniform(-r_max, r_max)
        y = np.random.uniform(-r_max, r_max)
        suma += calcular_contribucion_punto(x, y, r_max, rho_0)

    i_estimado = area * (suma / n_muestras)
    error_abs = abs(i_estimado - I_EXACTO)
    return i_estimado, error_abs


def monte_carlo_secuencial_numpy(
    n_muestras: int,
    r_max: float = R,
    rho_0: float = RHO_0,
    seed: Optional[int] = DEFAULT_SEED,
    chunk_size: int = 1_000_000,
) -> Dict[str, Any]:
    """
    Implementación secuencial vectorizada utilizando NumPy.
    Procesa en bloques (chunks) para optimizar el uso de memoria RAM y caché
    cuando N es grande (ej. N = 10^7 o 10^8).

    Parámetros:
    -----------
    n_muestras : int
        Número total de puntos aleatorios a generar.
    r_max : float
        Radio del disco (R = 1.0 por defecto).
    rho_0 : float
        Densidad superficial de referencia (rho_0 = 1.0 por defecto).
    seed : Optional[int]
        Semilla para el generador pseudoaleatorio de NumPy.
    chunk_size : int
        Tamaño máximo de lote procesado simultáneamente en memoria.

    Retorna:
    --------
    Dict[str, Any]:
        - 'n_muestras': N evaluado.
        - 'i_estimado': Aproximación Monte Carlo de la integral.
        - 'i_exacto': Valor analítico 0.9 * pi.
        - 'error_absoluto': |I_estimado - I_exacto|.
        - 'error_relativo_pct': Error porcentual relativo.
        - 'desviacion_estandar': Desviación estándar muestral s_f.
        - 'error_estandar_estimador': Error estadístico teórico s_f / sqrt(N).
        - 'fraccion_dentro_disco': Proporción de puntos con r <= R (aprox. pi/4).
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    area = 4.0 * (r_max ** 2)
    r_max_sq = r_max ** 2

    muestras_restantes = n_muestras
    suma_total = 0.0
    suma_cuadrados_total = 0.0
    puntos_dentro = 0

    while muestras_restantes > 0:
        actual_chunk = min(muestras_restantes, chunk_size)

        # Muestreo uniforme en [-R, R] x [-R, R]
        x = rng.uniform(-r_max, r_max, size=actual_chunk)
        y = rng.uniform(-r_max, r_max, size=actual_chunk)

        # Función indicadora del disco: r^2 <= R^2
        r2 = x * x + y * y
        mask = r2 <= r_max_sq
        puntos_dentro += int(np.count_nonzero(mask))

        # f(x, y) = rho(r) * r^2 * 1[r <= R]
        # Donde rho(r) = rho_0 * (1 + r / R)
        f_vals = np.zeros(actual_chunk, dtype=np.float64)
        if np.any(mask):
            r_inside = np.sqrt(r2[mask])
            rho_inside = rho_0 * (1.0 + r_inside / r_max)
            f_vals[mask] = rho_inside * r2[mask]

        suma_total += float(np.sum(f_vals))
        suma_cuadrados_total += float(np.sum(f_vals ** 2))
        muestras_restantes -= actual_chunk

    media_f = suma_total / n_muestras
    i_estimado = area * media_f
    error_abs = abs(i_estimado - I_EXACTO)
    error_rel_pct = (error_abs / I_EXACTO) * 100.0

    # Varianza y error estándar estadístico del estimador
    # Var(f) = E[f^2] - (E[f])^2
    var_f = max(0.0, (suma_cuadrados_total / n_muestras) - (media_f ** 2))
    std_f = math.sqrt(var_f)
    error_estandar = area * (std_f / math.sqrt(n_muestras))

    fraccion_dentro = puntos_dentro / n_muestras

    return {
        "n_muestras": n_muestras,
        "i_estimado": i_estimado,
        "i_exacto": I_EXACTO,
        "error_absoluto": error_abs,
        "error_relativo_pct": error_rel_pct,
        "desviacion_estandar": std_f,
        "error_estandar_estimador": error_estandar,
        "fraccion_dentro_disco": fraccion_dentro,
    }


if __name__ == "__main__":
    import json
    print(f"Valor analítico exacto I_exacto = {I_EXACTO:.8f}")
    res = monte_carlo_secuencial_numpy(100_000, seed=42)
    print(json.dumps(res, indent=2))
