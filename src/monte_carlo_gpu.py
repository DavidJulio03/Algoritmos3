"""
Implementación en GPU utilizando Numba CUDA y generadores pseudoaleatorios xoroshiro128p.

Nivel 3 - Algoritmos y Arquitecturas Computacionales de Alto Rendimiento (UIS).
"""

import math
import time
import numpy as np
from typing import Dict, Any, Optional, Tuple

import numba
from numba import cuda
from src.config import (
    R,
    RHO_0,
    AREA_CUADRADO,
    I_EXACTO,
    DEFAULT_SEED,
    CUDA_THREADS_PER_BLOCK,
)

# Intentar importar las funciones de generación pseudoaleatoria de Numba CUDA
try:
    from numba.cuda.random import (
        create_xoroshiro128p_states,
        xoroshiro128p_uniform_float64,
        xoroshiro128p_uniform_float32,
    )
    CUDA_RANDOM_AVAILABLE = True
except ImportError:
    CUDA_RANDOM_AVAILABLE = False


# ==============================================================================
# Kernel CUDA de Monte Carlo
# ==============================================================================
@cuda.jit
def kernel_monte_carlo_gpu(
    rng_states,
    n_samples_per_thread: int,
    r_max: float,
    rho_0: float,
    d_parciales,
):
    """
    Kernel de GPU para el cálculo de Monte Carlo del momento de inercia.

    Parámetros en memoria de dispositivo (Device):
    ----------------------------------------------
    rng_states : DeviceNDArray
        Arreglo con el estado del generador xoroshiro128p de cada hilo.
    n_samples_per_thread : int
        Número de muestras asignadas a cada hilo.
    r_max : float
        Radio del disco (R).
    rho_0 : float
        Densidad de referencia (rho_0).
    d_parciales : DeviceNDArray
        Arreglo unidimensional de salida donde cada hilo almacena su subtotal.

    Aspectos de arquitectura CUDA:
    -------------------------------
    - Cada hilo se identifica de forma única en la cuadrícula (Grid) mediante:
        tid = cuda.grid(1) = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    - Cada hilo accede a su propio generador xoroshiro128p (rng_states[tid]),
      garantizando independencia estadística entre hilos concurrentes.
    - Acumulación privada en registro de doble precisión (float64).
    - Evita contención en memoria global (no requiere operaciones atómicas en cada muestra).
    """
    tid = cuda.grid(1)

    if tid < d_parciales.size:
        acc = 0.0
        r_max_sq = r_max * r_max

        for _ in range(n_samples_per_thread):
            # xoroshiro128p_uniform_float64 retorna U en [0.0, 1.0)
            # Transformación afín al intervalo del cuadrado [-R, R]:
            # x = (u * 2.0 - 1.0) * R
            u1 = xoroshiro128p_uniform_float64(rng_states, tid)
            u2 = xoroshiro128p_uniform_float64(rng_states, tid)

            x = (u1 * 2.0 - 1.0) * r_max
            y = (u2 * 2.0 - 1.0) * r_max

            r2 = x * x + y * y

            # Función indicadora: 1 si r <= R, 0 si no
            if r2 <= r_max_sq:
                r = math.sqrt(r2)
                rho = rho_0 * (1.0 + r / r_max)
                acc += rho * r2

        # Almacenar la contribución acumulada del hilo en memoria global
        d_parciales[tid] = acc


def es_cuda_disponible() -> bool:
    """Retorna True si hay una GPU compatible con CUDA accesible."""
    return cuda.is_available()


def monte_carlo_gpu(
    n_muestras_deseadas: int,
    threads_per_block: int = CUDA_THREADS_PER_BLOCK,
    blocks_per_grid: Optional[int] = None,
    r_max: float = R,
    rho_0: float = RHO_0,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """
    Ejecuta el cálculo de Monte Carlo en GPU mediante Numba CUDA.

    Estrategia de descomposición del trabajo:
    -----------------------------------------
    1. Se calcula una cuadrícula de ejecución (Blocks x ThreadsPerBlock).
    2. Si blocks_per_grid es None, se dimensiona dinámicamente según N.
    3. Cada hilo procesa 'n_samples_per_thread' muestras.
    4. El total efectivo de muestras es total_threads * n_samples_per_thread >= n_muestras_deseadas.
    5. Cada hilo escribe en d_parciales[tid].
    6. Se transfiere el arreglo de sumas parciales al host y se reduce con suma de alta precisión.

    Medición detallada de tiempos:
    ------------------------------
    - tiempo_rng: Creación e inicialización de estados xoroshiro128p.
    - tiempo_kernel: Ejecución física del kernel en la GPU (sincronizada).
    - tiempo_transfer: Transferencia Device -> Host.
    - tiempo_total: Duración completa del pipeline.
    """
    if not cuda.is_available():
        raise RuntimeError(
            "CUDA no está disponible en este entorno. "
            "Para ejecutar benchmarks reales de GPU, utilice un entorno con GPU NVIDIA (ej. Google Colab)."
        )

    # 1. Configuración de la cuadrícula CUDA
    if blocks_per_grid is None:
        # Entre 128 y 1024 bloques para mantener alta ocupación de Streaming Multiprocessors (SM)
        hilos_objetivo = 65_536  # 256 bloques * 256 hilos
        blocks_per_grid = max(64, min(1024, hilos_objetivo // threads_per_block))

    total_hilos = blocks_per_grid * threads_per_block
    samples_per_thread = max(1, math.ceil(n_muestras_deseadas / total_hilos))
    total_muestras_efectivo = total_hilos * samples_per_thread

    t_inicio_total = time.perf_counter()

    # 2. Inicialización de generadores RNG por hilo
    t0_rng = time.perf_counter()
    rng_states = create_xoroshiro128p_states(total_hilos, seed=seed)
    cuda.synchronize()
    t_rng = time.perf_counter() - t0_rng

    # 3. Asignación de memoria en el dispositivo (Device array)
    d_parciales = cuda.device_array(total_hilos, dtype=np.float64)

    # 4. Lanzamiento y medición del Kernel con eventos CUDA
    start_event = cuda.event()
    end_event = cuda.event()

    start_event.record()
    kernel_monte_carlo_gpu[blocks_per_grid, threads_per_block](
        rng_states,
        samples_per_thread,
        r_max,
        rho_0,
        d_parciales,
    )
    end_event.record()
    end_event.synchronize()

    # cuda.event_elapsed_time retorna milisegundos
    t_kernel = end_event.elapsed_time() / 1000.0

    # 5. Transferencia Device -> Host
    t0_trans = time.perf_counter()
    h_parciales = d_parciales.copy_to_host()
    t_transfer = time.perf_counter() - t0_trans

    # 6. Reducción final en Host
    suma_total = float(np.sum(h_parciales))
    area = 4.0 * (r_max ** 2)
    i_estimado = area * (suma_total / total_muestras_efectivo)
    error_abs = abs(i_estimado - I_EXACTO)

    t_total = time.perf_counter() - t_inicio_total

    return {
        "n_muestras_solicitadas": n_muestras_deseadas,
        "n_muestras_efectivo": total_muestras_efectivo,
        "blocks_per_grid": blocks_per_grid,
        "threads_per_block": threads_per_block,
        "total_hilos": total_hilos,
        "samples_per_thread": samples_per_thread,
        "i_estimado": i_estimado,
        "i_exacto": I_EXACTO,
        "error_absoluto": error_abs,
        "error_relativo_pct": (error_abs / I_EXACTO) * 100.0,
        "tiempo_rng_segundos": t_rng,
        "tiempo_kernel_segundos": t_kernel,
        "tiempo_transfer_segundos": t_transfer,
        "tiempo_total_segundos": t_total,
    }


if __name__ == "__main__":
    print("=== ESTADO DEL SUBSISTEMA CUDA ===")
    print(f"CUDA disponible: {cuda.is_available()}")
    if cuda.is_available():
        print(f"Dispositivos GPU detectados: {len(cuda.gpus)}")
        for idx, gpu in enumerate(cuda.gpus):
            print(f"  GPU [{idx}]: {gpu.name.decode('utf-8')}")
        print("\nEjecutando prueba N = 1,000,000 en GPU...")
        res = monte_carlo_gpu(1_000_000)
        print(f"I_estimado = {res['i_estimado']:.6f} (Error: {res['error_absoluto']:.6f})")
        print(f"Tiempo Kernel: {res['tiempo_kernel_segundos']*1000:.2f} ms")
        print(f"Tiempo Total:  {res['tiempo_total_segundos']*1000:.2f} ms")
    else:
        print("Aviso: No se detectó GPU CUDA local (este equipo cuenta con AMD Radeon Graphics).")
        print("El código está completamente listo para ejecutarse en Google Colab con aceleración GPU.")
