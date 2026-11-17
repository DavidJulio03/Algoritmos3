"""
Pruebas unitarias para la implementación en GPU con Numba CUDA (Nivel 3).
"""

import math
import pytest
from src.config import R, RHO_0, I_EXACTO
from src.monte_carlo_gpu import (
    es_cuda_disponible,
    monte_carlo_gpu,
    kernel_monte_carlo_gpu,
)
from numba import cuda


def test_importacion_y_definicion_kernel_cuda():
    """Verifica que el kernel CUDA esté correctamente definido y decorado."""
    assert kernel_monte_carlo_gpu is not None
    # Comprobar que es un despachador CUDA
    assert hasattr(kernel_monte_carlo_gpu, "_func") or hasattr(kernel_monte_carlo_gpu, "forall")


def test_manejo_excepcion_sin_cuda():
    """
    Si no hay GPU NVIDIA local, verificar que la función lance una excepción clara
    en lugar de fallar de manera impredecible.
    """
    if not cuda.is_available():
        with pytest.raises(RuntimeError) as exc_info:
            monte_carlo_gpu(10_000)
        assert "CUDA no está disponible" in str(exc_info.value)


@pytest.mark.skipif(not cuda.is_available(), reason="Requiere hardware NVIDIA CUDA")
def test_ejecucion_kernel_cuda_real():
    """
    Prueba de integración para entornos con GPU real (ej. Google Colab).
    Verifica convergencia numérica y desglose de tiempos.
    """
    n = 200_000
    res = monte_carlo_gpu(n, seed=42)

    assert res["n_muestras_efectivo"] >= n
    assert math.isclose(res["i_exacto"], I_EXACTO, rel_tol=1e-12)
    assert res["error_relativo_pct"] < 2.0  # Menor al 2% de error

    # Comprobación de métricas de tiempo
    assert res["tiempo_kernel_segundos"] > 0
    assert res["tiempo_total_segundos"] >= res["tiempo_kernel_segundos"]
