"""
Script para generar el notebook de Google Colab (notebooks/monte_carlo_gpu_colab.ipynb).
"""

import json
from pathlib import Path

def create_colab_notebook():
    nb = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {
                "provenance": [],
                "toc_visible": True
            },
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3"
            },
            "language_info": {
                "name": "python"
            },
            "accelerator": "GPU"
        },
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Universidad Industrial de Santander (UIS)\n",
                    "## Algoritmos y Arquitecturas Computacionales de Alto Rendimiento\n",
                    "### Nivel 3 — Paralelización en GPU con Numba CUDA\n",
                    "**Problema:** Integración de Monte Carlo para el Momento de Inercia de un disco con densidad variable $\\rho(r) = \\rho_0(1 + r/R)$.\n",
                    "\n",
                    "> **INSTRUCCIÓN:** Para ejecutar este notebook en Google Colab, asegúrate de activar el acelerador por hardware:\n",
                    "> `Entorno de ejecución` -> `Cambiar tipo de entorno de ejecución` -> Seleccionar **T4 GPU**."
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Verificación del Hardware GPU (NVIDIA)"
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "!nvidia-smi\n",
                    "\n",
                    "import numba\n",
                    "from numba import cuda\n",
                    "print(f\"Numba version: {numba.__version__}\")\n",
                    "print(f\"CUDA disponible: {cuda.is_available()}\")\n",
                    "if cuda.is_available():\n",
                    "    for i, gpu in enumerate(cuda.gpus):\n",
                    "        print(f\"GPU [{i}]: {gpu.name.decode('utf-8')}\")\n",
                    "else:\n",
                    "    print(\"¡ADVERTENCIA! No se detectó GPU CUDA. Cambia el entorno de ejecución a GPU.\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Definición del Kernel CUDA y Funciones del Problema"
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "import math\n",
                    "import time\n",
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float64\n",
                    "\n",
                    "# Parámetros físicos\n",
                    "R = 1.0\n",
                    "RHO_0 = 1.0\n",
                    "AREA_CUADRADO = 4.0 * (R ** 2)\n",
                    "I_EXACTO = 0.9 * math.pi * RHO_0 * (R ** 4)\n",
                    "print(f\"Valor exacto analítico I_exacto = {I_EXACTO:.8f}\")\n",
                    "\n",
                    "@cuda.jit\n",
                    "def kernel_monte_carlo_gpu(rng_states, n_samples_per_thread, r_max, rho_0, d_parciales):\n",
                    "    \"\"\"\n",
                    "    Kernel CUDA: cada hilo genera coordenadas con xoroshiro128p, evalúa la función indicadora\n",
                    "    y acumula privadamente sus contribuciones en d_parciales[tid].\n",
                    "    \"\"\"\n",
                    "    tid = cuda.grid(1)\n",
                    "    if tid < d_parciales.size:\n",
                    "        acc = 0.0\n",
                    "        r_max_sq = r_max * r_max\n",
                    "        for _ in range(n_samples_per_thread):\n",
                    "            u1 = xoroshiro128p_uniform_float64(rng_states, tid)\n",
                    "            u2 = xoroshiro128p_uniform_float64(rng_states, tid)\n",
                    "            x = (u1 * 2.0 - 1.0) * r_max\n",
                    "            y = (u2 * 2.0 - 1.0) * r_max\n",
                    "            r2 = x * x + y * y\n",
                    "            if r2 <= r_max_sq:\n",
                    "                r = math.sqrt(r2)\n",
                    "                rho = rho_0 * (1.0 + r / r_max)\n",
                    "                acc += rho * r2\n",
                    "        d_parciales[tid] = acc"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Función de Ejecución y Medición de Tiempos Detallados"
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "def ejecutar_gpu_monte_carlo(n_deseado, threads_per_block=256, blocks_per_grid=256, seed=42):\n",
                    "    total_hilos = blocks_per_grid * threads_per_block\n",
                    "    samples_per_thread = max(1, math.ceil(n_deseado / total_hilos))\n",
                    "    n_efectivo = total_hilos * samples_per_thread\n",
                    "    \n",
                    "    t_start_total = time.perf_counter()\n",
                    "    \n",
                    "    # 1. Estados RNG\n",
                    "    t0_rng = time.perf_counter()\n",
                    "    rng_states = create_xoroshiro128p_states(total_hilos, seed=seed)\n",
                    "    cuda.synchronize()\n",
                    "    t_rng = time.perf_counter() - t0_rng\n",
                    "    \n",
                    "    # 2. Asignación de memoria en device\n",
                    "    d_parciales = cuda.device_array(total_hilos, dtype=np.float64)\n",
                    "    \n",
                    "    # 3. Kernel con eventos CUDA de alta precisión\n",
                    "    start_evt = cuda.event()\n",
                    "    end_evt = cuda.event()\n",
                    "    \n",
                    "    start_evt.record()\n",
                    "    kernel_monte_carlo_gpu[blocks_per_grid, threads_per_block](\n",
                    "        rng_states, samples_per_thread, R, RHO_0, d_parciales\n",
                    "    )\n",
                    "    end_evt.record()\n",
                    "    end_evt.synchronize()\n",
                    "    t_kernel = end_evt.elapsed_time() / 1000.0  # ms a segundos\n",
                    "    \n",
                    "    # 4. Transferencia Device -> Host\n",
                    "    t0_trans = time.perf_counter()\n",
                    "    h_parciales = d_parciales.copy_to_host()\n",
                    "    t_transfer = time.perf_counter() - t0_trans\n",
                    "    \n",
                    "    # 5. Reducción\n",
                    "    suma_total = float(np.sum(h_parciales))\n",
                    "    i_estimado = AREA_CUADRADO * (suma_total / n_efectivo)\n",
                    "    error_abs = abs(i_estimado - I_EXACTO)\n",
                    "    \n",
                    "    t_total = time.perf_counter() - t_start_total\n",
                    "    \n",
                    "    return {\n",
                    "        \"N\": n_deseado,\n",
                    "        \"N_efectivo\": n_efectivo,\n",
                    "        \"I_estimado\": i_estimado,\n",
                    "        \"error_absoluto\": error_abs,\n",
                    "        \"t_rng\": t_rng,\n",
                    "        \"t_kernel\": t_kernel,\n",
                    "        \"t_transfer\": t_transfer,\n",
                    "        \"t_total\": t_total,\n",
                    "    }\n",
                    "\n",
                    "# Warm-up JIT obligatorio para excluir tiempo de compilación\n",
                    "print(\"Ejecutando calentamiento JIT en GPU...\")\n",
                    "ejecutar_gpu_monte_carlo(10_000)\n",
                    "print(\"Calentamiento completado.\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Benchmarks de GPU para $N = 10^4, 10^5, 10^6, 10^7, 10^8$"
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "n_valores = [10**4, 10**5, 10**6, 10**7, 10**8]\n",
                    "repeticiones = 5\n",
                    "resultados_gpu = []\n",
                    "\n",
                    "for n in n_valores:\n",
                    "    print(f\"Evaluando GPU para N = {n:,}...\")\n",
                    "    corridas_n = []\n",
                    "    for r in range(repeticiones):\n",
                    "        res = ejecutar_gpu_monte_carlo(n, seed=100 + r)\n",
                    "        corridas_n.append(res)\n",
                    "    \n",
                    "    # Calcular mediana de las 5 corridas\n",
                    "    med_kernel = np.median([c[\"t_kernel\"] for c in corridas_n])\n",
                    "    med_total = np.median([c[\"t_total\"] for c in corridas_n])\n",
                    "    med_rng = np.median([c[\"t_rng\"] for c in corridas_n])\n",
                    "    med_trans = np.median([c[\"t_transfer\"] for c in corridas_n])\n",
                    "    i_est = corridas_n[0][\"I_estimado\"]\n",
                    "    err_abs = corridas_n[0][\"error_absoluto\"]\n",
                    "    \n",
                    "    resultados_gpu.append({\n",
                    "        \"N\": n,\n",
                    "        \"I_estimado_GPU\": i_est,\n",
                    "        \"Error_absoluto_GPU\": err_abs,\n",
                    "        \"T_kernel_seg\": med_kernel,\n",
                    "        \"T_total_seg\": med_total,\n",
                    "        \"T_rng_seg\": med_rng,\n",
                    "        \"T_transfer_seg\": med_trans,\n",
                    "    })\n",
                    "    print(f\"  [OK] N={n:,} | T_kernel={med_kernel*1000:.2f} ms | T_total={med_total*1000:.2f} ms | I_est={i_est:.6f}\")\n",
                    "\n",
                    "df_gpu = pd.DataFrame(resultados_gpu)\n",
                    "df_gpu.to_csv(\"tabla_benchmarks_gpu.csv\", index=False)\n",
                    "print(\"\\nResultados de GPU:\")\n",
                    "print(df_gpu.to_string(index=False))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Comparativa de Rendimiento y Speedup: GPU vs CPU"
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "# Datos medidos en CPU (Hardware AMD Zen 8 hilos):\n",
                    "tiempos_cpu = pd.DataFrame({\n",
                    "    \"N\": [10000, 100000, 1000000, 10000000, 100000000],\n",
                    "    \"Numba_Serial\": [0.000465, 0.004079, 0.035185, 0.361734, 3.610525],\n",
                    "    \"Numba_Paralelo\": [0.000139, 0.000684, 0.007451, 0.067689, 0.702136],\n",
                    "})\n",
                    "\n",
                    "comparativa = pd.merge(df_gpu, tiempos_cpu, on=\"N\")\n",
                    "comparativa[\"Speedup_GPU_Kernel_vs_CPU_Serial\"] = comparativa[\"Numba_Serial\"] / comparativa[\"T_kernel_seg\"]\n",
                    "comparativa[\"Speedup_GPU_Total_vs_CPU_Serial\"] = comparativa[\"Numba_Serial\"] / comparativa[\"T_total_seg\"]\n",
                    "comparativa[\"Speedup_GPU_Kernel_vs_CPU_Paralelo\"] = comparativa[\"Numba_Paralelo\"] / comparativa[\"T_kernel_seg\"]\n",
                    "\n",
                    "comparativa.to_csv(\"comparativa_rendimiento_global.csv\", index=False)\n",
                    "print(comparativa[[\"N\", \"Numba_Serial\", \"Numba_Paralelo\", \"T_kernel_seg\", \"T_total_seg\", \"Speedup_GPU_Kernel_vs_CPU_Serial\", \"Speedup_GPU_Kernel_vs_CPU_Paralelo\"]].to_string(index=False))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. Gráfica Comparativa de Speedup"
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "plt.figure(figsize=(10, 6), dpi=300)\n",
                    "plt.plot(comparativa[\"N\"], comparativa[\"Numba_Serial\"] / comparativa[\"Numba_Paralelo\"], marker=\"o\", linewidth=2, label=\"Speedup CPU Paralelo (8 hilos)\")\n",
                    "plt.plot(comparativa[\"N\"], comparativa[\"Speedup_GPU_Kernel_vs_CPU_Serial\"], marker=\"^\", linewidth=2.2, color=\"#d62728\", label=\"Speedup GPU Kernel (vs CPU Serial)\")\n",
                    "plt.plot(comparativa[\"N\"], comparativa[\"Speedup_GPU_Total_vs_CPU_Serial\"], marker=\"s\", linewidth=1.8, linestyle=\"--\", color=\"#9467bd\", label=\"Speedup GPU Total (vs CPU Serial)\")\n",
                    "plt.axhline(1.0, color=\"gray\", linestyle=\":\", label=\"Umbral S = 1.0\")\n",
                    "\n",
                    "plt.xscale(\"log\")\n",
                    "plt.xlabel(\"Número de muestras ($N$)\", fontsize=12, fontweight=\"bold\")\n",
                    "plt.ylabel(\"Speedup ($S$)\", fontsize=12, fontweight=\"bold\")\n",
                    "plt.title(\"Comparativa Global de Speedup: CPU Paralelo vs GPU CUDA\", fontsize=13, pad=12)\n",
                    "plt.grid(True, which=\"both\", linestyle=\":\", alpha=0.6)\n",
                    "plt.legend(fontsize=10, loc=\"upper left\")\n",
                    "plt.tight_layout()\n",
                    "plt.savefig(\"speedup_comparativo_global.png\")\n",
                    "plt.show()"
                ]
            }
        ]
    }
    
    out_path = Path("notebooks/monte_carlo_gpu_colab.ipynb")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"Notebook Colab generado exitosamente en: {out_path}")

if __name__ == "__main__":
    create_colab_notebook()
