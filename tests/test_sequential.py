"""
Pruebas unitarias para la implementación secuencial (Nivel 1).
"""

import math
import pytest
from src.config import R, RHO_0, AREA_CUADRADO, I_EXACTO, DEFAULT_SEED
from src.monte_carlo_sequential import (
    calcular_contribucion_punto,
    monte_carlo_secuencial_puro,
    monte_carlo_secuencial_numpy,
)


def test_constantes_fisicas_y_analiticas():
    """Verifica que las constantes fundamentales del problema sean exactas."""
    assert R == 1.0
    assert RHO_0 == 1.0
    assert AREA_CUADRADO == 4.0 * (R ** 2)
    assert math.isclose(I_EXACTO, 0.9 * math.pi, rel_tol=1e-12)


def test_evaluacion_contribucion_puntos_clave():
    """
    Verifica que la función indicadora y la densidad se evalúen con precisión:
    - En el origen: r = 0, rho = 1, r^2 = 0 -> contrib = 0.0
    - En el borde (1, 0): r = 1, rho = 1*(1+1) = 2, r^2 = 1 -> contrib = 2.0
    - Fuera del disco (1.1, 0): r > 1 -> contrib = 0.0
    - En la esquina del cuadrado (1, 1): r = sqrt(2) > 1 -> contrib = 0.0
    - Punto interior (0.5, 0): r = 0.5, rho = 1.5, r^2 = 0.25 -> contrib = 0.375
    """
    # Origen
    assert calcular_contribucion_punto(0.0, 0.0, R, RHO_0) == 0.0

    # Borde exacto r = 1
    assert math.isclose(calcular_contribucion_punto(1.0, 0.0, R, RHO_0), 2.0, rel_tol=1e-9)
    assert math.isclose(calcular_contribucion_punto(0.0, -1.0, R, RHO_0), 2.0, rel_tol=1e-9)

    # Fuera del disco
    assert calcular_contribucion_punto(1.001, 0.0, R, RHO_0) == 0.0
    assert calcular_contribucion_punto(1.0, 1.0, R, RHO_0) == 0.0
    assert calcular_contribucion_punto(-0.8, -0.8, R, RHO_0) == 0.0  # r^2 = 1.28 > 1

    # Punto interior conocido
    # r = 0.6, rho = 1.6, r^2 = 0.36 -> contrib = 1.6 * 0.36 = 0.576
    assert math.isclose(calcular_contribucion_punto(0.6, 0.0, R, RHO_0), 0.576, rel_tol=1e-9)


def test_reproducibilidad_con_semilla():
    """Garantiza que la misma semilla genere idénticos resultados."""
    res1 = monte_carlo_secuencial_numpy(10_000, seed=123)
    res2 = monte_carlo_secuencial_numpy(10_000, seed=123)
    assert res1["i_estimado"] == res2["i_estimado"]
    assert res1["error_absoluto"] == res2["error_absoluto"]


def test_consistencia_puro_vs_numpy():
    """Comprueba que la versión pura y la vectorizada sean estadísticamente coherentes."""
    n = 20_000
    i_puro, err_puro = monte_carlo_secuencial_puro(n, seed=42)
    res_np = monte_carlo_secuencial_numpy(n, seed=42)

    # Ambos deben aproximar el valor exacto dentro de un margen razonable (< 5%)
    assert abs(i_puro - I_EXACTO) < 0.15
    assert abs(res_np["i_estimado"] - I_EXACTO) < 0.15


def test_fraccion_puntos_dentro_del_disco():
    """
    La fracción de puntos dentro del disco debe aproximar pi * R^2 / (4 * R^2) = pi / 4 ~ 0.785398
    """
    res = monte_carlo_secuencial_numpy(100_000, seed=999)
    esperado = math.pi / 4.0
    assert math.isclose(res["fraccion_dentro_disco"], esperado, rel_tol=0.01)


def test_error_dentro_de_intervalo_estadistico():
    """
    El error absoluto debe ser menor a 3 veces el error estándar estimado (criterio 3-sigma: 99.7% confianza).
    """
    res = monte_carlo_secuencial_numpy(200_000, seed=777)
    limite_3sigma = 3.0 * res["error_estandar_estimador"]
    assert res["error_absoluto"] <= limite_3sigma
