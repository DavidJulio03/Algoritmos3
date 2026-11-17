"""
Módulo de benchmarks rigurosos de rendimiento y análisis de Speedup (Nivel 2).
Compara la versión secuencial y la paralela para N = 10^4, 10^5, ..., 10^8.
Genera tablas de datos y gráficas de Speedup y tiempos de ejecución.
"""

import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any

from src.config import (
    N_VALORES_BENCHMARKS,
    BENCHMARK_REPETICIONES,
    DEFAULT_SEED,
    RAW_RESULTS_DIR,
    TABLES_DIR,
    FIGURES_DIR,
    I_EXACTO,
)
from src.monte_carlo_sequential import monte_carlo_secuencial_numpy
from src.monte_carlo_cpu import (
    calentar_numba_cpu,
    monte_carlo_cpu_serial,
    monte_carlo_cpu_parallel,
)
import numba


def ejecutar_benchmarks_cpu(
    n_valores: List[int] = N_VALORES_BENCHMARKS,
    repeticiones: int = BENCHMARK_REPETICIONES,
    seed_base: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """
    Ejecuta mediciones rigurosas de tiempo para versiones secuenciales y paralelas.

    Aspectos metodológicos:
    1. Calentamiento JIT previo (warm-up) para excluir tiempos de compilación.
    2. Mediciones con time.perf_counter() de alta resolución.
    3. Múltiples repeticiones para calcular la mediana y mitigar ruido del SO.
    """
    print("=" * 75)
    print("INICIANDO BENCHMARKS DE RENDIMIENTO EN CPU (NIVEL 2)")
    print(f"Hilos disponibles en hardware/Numba: {numba.get_num_threads()}")
    print(f"Configuraciones de N: {n_valores}")
    print(f"Repeticiones por configuración: {repeticiones}")
    print("=" * 75)

    print("\nEjecutando calentamiento JIT...")
    calentar_numba_cpu()
    # Calentamiento rápido de numpy
    monte_carlo_secuencial_numpy(1_000, seed=1)
    print("Calentamiento completado exitosamente.\n")

    filas = []

    for n in n_valores:
        print(f"--- Evaluando N = {n:,} ---")

        # 1. Secuencial Numba (Serial)
        tiempos_numba_serial = []
        for r in range(repeticiones):
            t0 = time.perf_counter()
            res = monte_carlo_cpu_serial(n, seed=seed_base + r)
            t_ejec = time.perf_counter() - t0
            tiempos_numba_serial.append(t_ejec)
            filas.append({
                "N": n,
                "repeticion": r + 1,
                "implementacion": "Numba_Serial",
                "tiempo_segundos": t_ejec,
                "I_estimado": res["i_estimado"],
                "error_absoluto": res["error_absoluto"],
            })

        # 2. Paralelo Numba (Multi-hilo)
        tiempos_numba_paralelo = []
        for r in range(repeticiones):
            t0 = time.perf_counter()
            res = monte_carlo_cpu_parallel(n, seed=seed_base + r)
            t_ejec = time.perf_counter() - t0
            tiempos_numba_paralelo.append(t_ejec)
            filas.append({
                "N": n,
                "repeticion": r + 1,
                "implementacion": "Numba_Paralelo",
                "tiempo_segundos": t_ejec,
                "I_estimado": res["i_estimado"],
                "error_absoluto": res["error_absoluto"],
            })

        # 3. Secuencial NumPy (Vectorizado Nivel 1)
        # Para N=10^8 ejecutamos 3 repeticiones si repeticiones > 3 para optimizar tiempo
        reps_numpy = min(repeticiones, 3) if n >= 10**8 else repeticiones
        for r in range(reps_numpy):
            t0 = time.perf_counter()
            res = monte_carlo_secuencial_numpy(n, seed=seed_base + r)
            t_ejec = time.perf_counter() - t0
            filas.append({
                "N": n,
                "repeticion": r + 1,
                "implementacion": "NumPy_Secuencial",
                "tiempo_segundos": t_ejec,
                "I_estimado": res["i_estimado"],
                "error_absoluto": res["error_absoluto"],
            })

        med_s = np.median(tiempos_numba_serial)
        med_p = np.median(tiempos_numba_paralelo)
        sp = med_s / med_p if med_p > 0 else 0
        print(f"  [OK] N={n:,} | T_ser={med_s*1000:8.2f} ms | T_par={med_p*1000:8.2f} ms | Speedup={sp:.2f}x")

    df = pd.DataFrame(filas)
    RAW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_RESULTS_DIR / "benchmarks_cpu_crudo.csv"
    df.to_csv(raw_path, index=False)
    print(f"\nDatos brutos de benchmarks guardados en: {raw_path}")

    return df


def procesar_tabla_speedup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula medianas, desviaciones estándar y factores de aceleración (Speedup).
    """
    # Agrupar por N e implementación
    agrupado = df.groupby(["N", "implementacion"])["tiempo_segundos"].agg(
        mediana="median",
        media="mean",
        std="std"
    ).reset_index()

    pivote = agrupado.pivot(index="N", columns="implementacion", values="mediana").reset_index()

    # Speedup = T_secuencial / T_paralelo
    pivote["speedup_numba"] = pivote["Numba_Serial"] / pivote["Numba_Paralelo"]
    pivote["speedup_vs_numpy"] = pivote["NumPy_Secuencial"] / pivote["Numba_Paralelo"]

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    tabla_path = TABLES_DIR / "tabla_speedup_cpu.csv"
    pivote.to_csv(tabla_path, index=False)
    print(f"Tabla de speedup procesada guardada en: {tabla_path}")

    return pivote


def generar_graficas_rendimiento_cpu(df_resumen: pd.DataFrame) -> None:
    """
    Genera dos gráficas científicas:
    1. Speedup en función de N (semilogx).
    2. Tiempos de ejecución en función de N (log-log).
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    num_hilos = numba.get_num_threads()

    # --- Gráfica 1: Speedup vs N ---
    fig_sp_path = FIGURES_DIR / "speedup_cpu.png"
    plt.figure(figsize=(9, 6), dpi=300)

    plt.plot(
        df_resumen["N"],
        df_resumen["speedup_numba"],
        marker="o",
        color="#2ca02c",
        linewidth=2.4,
        markersize=8,
        label=r"Speedup CPU ($T_{Numba\_Serial} / T_{Numba\_Paralelo}$)",
        zorder=4,
    )

    # Línea de referencia ideal (Speedup teórico lineal = num_hilos)
    plt.axhline(
        y=num_hilos,
        color="#d62728",
        linestyle="--",
        linewidth=1.8,
        label=f"Límite teórico ideal ({num_hilos} hilos)",
        zorder=2,
    )

    # Línea de referencia de no-ganancia (Speedup = 1.0)
    plt.axhline(
        y=1.0,
        color="#7f7f7f",
        linestyle=":",
        linewidth=1.5,
        label="Umbral sin ganancia (S = 1.0)",
        zorder=2,
    )

    plt.xscale("log")
    plt.xlabel("Número de muestras ($N$)", fontsize=12, fontweight="bold")
    plt.ylabel("Speedup ($S = T_{sec} / T_{par}$)", fontsize=12, fontweight="bold")
    plt.title(
        f"Aceleración (Speedup) en CPU con Numba vs Tamaño de Muestra\n"
        f"Arquitectura: AMD Zen (4 Núcleos / 8 Hilos de Hardware)",
        fontsize=13,
        pad=12,
    )
    plt.ylim(0, num_hilos + 1.5)
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend(fontsize=10, loc="upper left", framealpha=0.95)
    plt.tight_layout()
    plt.savefig(fig_sp_path)
    plt.close()
    print(f"Gráfica de Speedup guardada en: {fig_sp_path}")

    # --- Gráfica 2: Tiempos de Ejecución vs N (log-log) ---
    fig_time_path = FIGURES_DIR / "tiempos_ejecucion_cpu.png"
    plt.figure(figsize=(9, 6), dpi=300)

    plt.plot(
        df_resumen["N"],
        df_resumen["NumPy_Secuencial"],
        marker="s",
        color="#1f77b4",
        linewidth=2.0,
        markersize=7,
        label="NumPy Secuencial (Vectorizado Nivel 1)",
    )
    plt.plot(
        df_resumen["N"],
        df_resumen["Numba_Serial"],
        marker="^",
        color="#ff7f0e",
        linewidth=2.0,
        markersize=7,
        label="Numba Serial (1 hilo CPU)",
    )
    plt.plot(
        df_resumen["N"],
        df_resumen["Numba_Paralelo"],
        marker="o",
        color="#2ca02c",
        linewidth=2.2,
        markersize=8,
        label=f"Numba Paralelo ({num_hilos} hilos CPU)",
    )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Número de muestras ($N$)", fontsize=12, fontweight="bold")
    plt.ylabel("Tiempo de ejecución (segundos)", fontsize=12, fontweight="bold")
    plt.title(
        "Tiempo de Ejecución vs Tamaño de Muestra (Escala Log-Log)\n"
        "Comparativa de Implementaciones en CPU",
        fontsize=13,
        pad=12,
    )
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend(fontsize=10, loc="upper left", framealpha=0.95)
    plt.tight_layout()
    plt.savefig(fig_time_path)
    plt.close()
    print(f"Gráfica de Tiempos guardada en: {fig_time_path}")


def ejecutar_flujo_benchmarks():
    """Ejecuta benchmarks, procesa tabla de speedup y genera las gráficas."""
    df_crudo = ejecutar_benchmarks_cpu()
    resumen = procesar_tabla_speedup(df_crudo)
    generar_graficas_rendimiento_cpu(resumen)
    print("\n" + "=" * 75)
    print("RESUMEN DE BENCHMARKS Y SPEEDUP:")
    print("=" * 75)
    print(resumen.to_string(index=False))
    return df_crudo, resumen


if __name__ == "__main__":
    ejecutar_flujo_benchmarks()
