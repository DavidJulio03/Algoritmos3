"""
Pruebas unitarias para la paralelización en CPU con Numba (Nivel 2).
"""

import math
import pytest
import numba
from src.config import R, RHO_0, I_EXACTO
from src.monte_carlo_cpu import (
    calentar_numba_cpu,
    monte_carlo_cpu_serial,
    monte_carlo_cpu_parallel,
)


@pytest.fixture(scope="module", autouse=True)
def setup_warmup():
    """Ejecuta el calentamiento JIT antes de las pruebas del módulo."""
    calentar_numba_cpu()


def test_hilos_disponibles_numba():
    """Verifica que Numba reconozca los hilos de hardware disponibles."""
    num_hilos = numba.get_num_threads()
    assert num_hilos >= 1
    print(f"\n[INFO] Hilos activos de Numba: {num_hilos}")


def test_correccion_matematica_serial_numba():
    """Verifica que el kernel serial converja hacia I_exacto = 0.9 * pi."""
    n = 200_000
    res = monte_carlo_cpu_serial(n, seed=1234)
    assert math.isclose(res["i_exacto"], I_EXACTO, rel_tol=1e-12)
    assert res["error_relativo_pct"] < 2.0  # Menor al 2% para N=200,000


def test_correccion_matematica_paralelo_numba():
    """Verifica que el kernel paralelo con prange converja hacia I_exacto."""
    n = 200_000
    res = monte_carlo_cpu_parallel(n, seed=5678)
    assert math.isclose(res["i_exacto"], I_EXACTO, rel_tol=1e-12)
    assert res["error_relativo_pct"] < 2.0


def test_consistencia_serial_vs_paralelo():
    """
    Comprueba que la diferencia entre la estimación serial y paralela
    esté dentro del orden de magnitud del error estadístico de Monte Carlo.
    """
    n = 500_000
    res_s = monte_carlo_cpu_serial(n, seed=101)
    res_p = monte_carlo_cpu_parallel(n, seed=202)

    diferencia = abs(res_s["i_estimado"] - res_p["i_estimado"])
    # Para N=500,000 el error estadístico es ~ 0.003
    assert diferencia < 0.03


def test_reduccion_paralela_sin_condicion_de_carrera():
    """
    Comprueba que múltiples ejecuciones paralelas no corrompan el acumulador
    ni generen valores absurdos, NaN o infinitos por condiciones de carrera.
    """
    n = 100_000
    for s in [10, 20, 30]:
        res = monte_carlo_cpu_parallel(n, seed=s)
        assert not math.isnan(res["i_estimado"])
        assert not math.isinf(res["i_estimado"])
        # El valor físico del momento de inercia debe estar acotado:
        # Menor que una masa total con radio 1 y densidad max: I < 4 * 2 * 1 = 8
        assert 1.5 < res["i_estimado"] < 4.0
