"""
Módulo de validación científica y análisis de convergencia de Monte Carlo (Nivel 1).
Ejecuta los experimentos para N = 10^3, 10^4, 10^5, 10^6, 10^7,
guarda resultados en CSV y genera la gráfica de convergencia en escala log-log.
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any

from src.config import (
    N_VALORES_NIVEL_1,
    I_EXACTO,
    R,
    RHO_0,
    AREA_CUADRADO,
    DEFAULT_SEED,
    RAW_RESULTS_DIR,
    TABLES_DIR,
    FIGURES_DIR,
)
from src.monte_carlo_sequential import monte_carlo_secuencial_numpy


def ejecutar_experimento_convergencia_nivel1(
    n_valores: List[int] = N_VALORES_NIVEL_1,
    num_semillas: int = 10,
    seed_base: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """
    Ejecuta el experimento de convergencia para múltiples tamaños de muestra N
    y múltiples semillas aleatorias independientes para capturar la variabilidad estadística.

    Parámetros:
    -----------
    n_valores : List[int]
        Lista de tamaños de muestra N (ej. 10^3, 10^4, ..., 10^7).
    num_semillas : int
        Número de repeticiones independientes por cada N.
    seed_base : int
        Semilla inicial para generar las semillas secundarias.

    Retorna:
    --------
    pd.DataFrame con todas las corridas registradas.
    """
    filas = []
    rng_master = np.random.default_rng(seed_base)
    semillas = [int(s) for s in rng_master.integers(1, 1_000_000, size=num_semillas)]

    print("=" * 70)
    print("EJECUTANDO EXPERIMENTOS DE CONVERGENCIA - NIVEL 1")
    print(f"Valores de N: {n_valores}")
    print(f"Número de repeticiones por N: {num_semillas}")
    print(f"Valor analítico exacto I_exacto: {I_EXACTO:.8f}")
    print("=" * 70)

    for n in n_valores:
        for idx_run, s in enumerate(semillas):
            res = monte_carlo_secuencial_numpy(n, seed=s)
            filas.append({
                "N": n,
                "corrida": idx_run + 1,
                "semilla": s,
                "I_estimado": res["i_estimado"],
                "I_exacto": res["i_exacto"],
                "error_absoluto": res["error_absoluto"],
                "error_relativo_pct": res["error_relativo_pct"],
                "error_estandar_estimador": res["error_estandar_estimador"],
                "fraccion_dentro_disco": res["fraccion_dentro_disco"],
            })
        print(f"  [OK] Completado N = {n:,} ({num_semillas} repeticiones)")

    df = pd.DataFrame(filas)

    # Guardar datos brutos
    RAW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_RESULTS_DIR / "nivel1_convergencia_crudo.csv"
    df.to_csv(raw_path, index=False)
    print(f"\nDatos brutos guardados en: {raw_path}")

    return df


def procesar_resumen_convergencia(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa los datos por N y calcula estadísticas representativas:
    media, mediana, desviación estándar del error absoluto y error relativo.
    """
    resumen = df.groupby("N").agg(
        I_estimado_media=("I_estimado", "mean"),
        I_estimado_std=("I_estimado", "std"),
        error_abs_media=("error_absoluto", "mean"),
        error_abs_mediana=("error_absoluto", "median"),
        error_abs_std=("error_absoluto", "std"),
        error_rel_pct_media=("error_relativo_pct", "mean"),
        error_estandar_teorico=("error_estandar_estimador", "mean"),
    ).reset_index()

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    tabla_path = TABLES_DIR / "tabla_convergencia_nivel1.csv"
    resumen.to_csv(tabla_path, index=False)
    print(f"Tabla de resumen guardada en: {tabla_path}")

    return resumen


def generar_grafica_convergencia(df: pd.DataFrame, resumen: pd.DataFrame) -> Path:
    """
    Genera una figura científica en escala log-log comparando:
    1. Errores absolutos individuales de cada corrida.
    2. Mediana del error empírico vs N.
    3. Recta de referencia teórica proporcional a N^(-1/2) (pendiente -1/2).
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIGURES_DIR / "convergencia_error_nivel1.png"

    plt.figure(figsize=(9, 6), dpi=300)

    # Dispersión de las corridas individuales
    plt.scatter(
        df["N"],
        df["error_absoluto"],
        color="#7f7f7f",
        alpha=0.45,
        s=30,
        label="Corridas individuales (semillas independientes)",
        zorder=2,
    )

    # Mediana del error absoluto por cada N
    plt.plot(
        resumen["N"],
        resumen["error_abs_mediana"],
        marker="o",
        color="#1f77b4",
        linewidth=2.2,
        markersize=7,
        label=r"Mediana empírica $|I_{estimado} - I_{exacto}|$",
        zorder=4,
    )

    # Recta de referencia teórica C * N^(-1/2) (pendiente -1/2 en escala log-log)
    # Ajustamos la constante C para que coincida en escala con la primera mediana
    c_ref = float(resumen["error_abs_mediana"].iloc[0] * math.sqrt(resumen["N"].iloc[0]))
    n_linea = np.logspace(np.log10(resumen["N"].min()), np.log10(resumen["N"].max()), 100)
    error_ref = c_ref / np.sqrt(n_linea)

    plt.plot(
        n_linea,
        error_ref,
        linestyle="--",
        color="#d62728",
        linewidth=2.0,
        label=r"Pendiente teórica $\propto N^{-1/2}$ (Convergencia Monte Carlo)",
        zorder=3,
    )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Número de muestras ($N$)", fontsize=12, fontweight="bold")
    plt.ylabel("Error absoluto $|I - I_{exacto}|$", fontsize=12, fontweight="bold")
    plt.title(
        "Convergencia del Estimador de Monte Carlo (Nivel 1)\n"
        r"Momento de Inercia del Disco: $I_{exacto} = 0.9\pi \approx 2.827433$",
        fontsize=13,
        pad=12,
    )
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend(fontsize=10, loc="upper right", framealpha=0.95)
    plt.tight_layout()

    plt.savefig(fig_path)
    plt.close()

    print(f"Gráfica de convergencia generada exitosamente en: {fig_path}")
    return fig_path


def ejecutar_flujo_completo_nivel1():
    """Ejecuta experimentos, genera tablas y la gráfica oficial."""
    df_crudo = ejecutar_experimento_convergencia_nivel1()
    resumen = procesar_resumen_convergencia(df_crudo)
    fig_path = generar_grafica_convergencia(df_crudo, resumen)
    print("\nResumen de resultados Nivel 1:")
    print(resumen.to_string(index=False))
    return df_crudo, resumen, fig_path


if __name__ == "__main__":
    ejecutar_flujo_completo_nivel1()
