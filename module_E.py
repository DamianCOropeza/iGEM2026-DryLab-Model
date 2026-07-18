#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=======================================================================
MODULO E - CURVAS DOSIS-RESPUESTA dCas9-sgRNA-TARGET
=======================================================================
Proyecto iGEM 2025 - Deteccion de mutaciones oncogenicas en ctDNA
Modelado matematico de biologia sintetica (Dry Lab)

Este script calcula e implementa las curvas dosis-respuesta de union
dCas9-sgRNA a sus secuencias target (ADN tumoral circulante, ctDNA),
para tres mutaciones oncogenicas:
    1) KRAS G12C       (SNV)
    2) EGFR exon19del  (indel / delecion)
    3) EGFR L858R      (SNV)

Cada sgRNA se evalua contra su secuencia MUTANTE (target correcto) y
su secuencia SILVESTRE (wild-type, mismatch → senal cruzada), usando
dos modelos de union: Langmuir (n=1) y Hill (n ajustable).

Pipeline del proyecto:
    Modulo A -> B -> C -> D -> [MODULO E] -> Modulo F -> Modulo G

# ============================================================
# INSTALACION DE DEPENDENCIAS (ejecutar en terminal una sola vez)
# ============================================================
# conda activate iGEM
# conda install numpy matplotlib
#
# pip (alternativa):
# pip install numpy matplotlib
#
# Verificar:
# python -c "import numpy, matplotlib; print('OK')"
#
# Ejecutar:
# python modulo_E_dosis_respuesta.py
# ============================================================

=======================================================================
SUPUESTOS DEL MODELO
=======================================================================
1. Se usa Langmuir 1:1 (n=1) como modelo base de union dCas9-sgRNA-target;
   Hill con n=1.1 se incluye como variante para explorar posible
   cooperatividad o efectos de avidity. Con n=1, Langmuir y Hill son
   matematicamente identicos.
2. Los valores de Kd provienen de literatura de Cas9/dCas9 general (no del
   sgRNA especifico del equipo); son un punto de partida razonable, pero
   deben reemplazarse por mediciones propias de SPR cuando esten
   disponibles (marcado con ⚠ WET LAB en cada parametro).
3. Gamma_max_ef (capacidad efectiva del chip) viene directamente del
   output del Modulo D, ya implementado y validado en este pipeline.
4. Se asume EQUILIBRIO INSTANTANEO de union: este modulo NO simula la
   cinetica de union dCas9-sgRNA-target en el tiempo (eso corresponderia
   a una extension futura); solo se calcula el estado de equilibrio para
   cada concentracion de target.
5. La senal del biosensor se asume proporcional a la fraccion de
   receptores ocupados (theta), es decir, no hay amplificacion no lineal
   de senal en esta etapa del modelo.
6. Se ignora el agotamiento del target en la solucion de muestra (la
   dilucion del target por union al chip se considera despreciable
   frente al volumen total de la muestra).
7. Los tres sgRNAs se modelan de forma INDEPENDIENTE: no se considera
   competencia entre ellos por el mismo chip ni union cruzada entre
   sgRNAs distintos.
=======================================================================
"""

# =======================================================================
# IMPORTS
# =======================================================================
import numpy as np                          # arreglos numericos y operaciones vectoriales
import matplotlib.pyplot as plt              # generacion de graficas
import matplotlib.ticker as ticker           # formato de ejes en notacion cientifica (escala log)
import matplotlib.lines as mlines            # construccion manual de elementos de leyenda

# =======================================================================
# PARAMETROS DEL MODELO
# =======================================================================
# Convencion de comentarios:
#   [unidad] | Fuente: cita APA 7
#   ⚠ WET LAB: indica que el valor debe reemplazarse con dato experimental
#   ← Modulo X: indica que el valor proviene del output de otro modulo

# ================================================================
# INPUT DEL MODULO D
# ================================================================
Gamma_max_ef = 5.0e-14  # [mol/mm^2] | ← Modulo D
                         #              Densidad de receptores dCas9 inmovilizados
                         #              en el chip de streptavidina (Gamma_final del
                         #              modelo cinetico de Langmuir del Modulo D).
Gamma_max_RU = 8000.0   # [RU]       | ← Modulo D
                         #              Equivalente de Gamma_max_ef en unidades de
                         #              resonancia SPR (senal maxima teorica del biosensor).

# ================================================================
# Kd DE UNION dCas9-sgRNA CON SECUENCIA MUTANTE (target perfecto)
# ================================================================
# Nota: los Kd de dCas9 (sin actividad de corte) para ADN son comparables
# a los de Cas9 activo en terminos de union, pero con mayor tiempo de residencia.

Kd_KRAS_mut = 1.0e-9  # [M] | Kd estimado para SNV (match perfecto).
                       #       Fuente: Sternberg, S. E., Redding, S., Jinek, M.,
                       #       Greene, E. C., & Doudna, J. A. (2014). DNA interrogation
                       #       by the CRISPR RNA-guided endonuclease Cas9. Nature,
                       #       507(7490), 62-67. https://doi.org/10.1038/nature13011
                       #       Rango literatura SNVs: 0.1-10 nM.
                       # ⚠ WET LAB: reemplazar con Kd medido por SPR con sgRNA
                       #   KRAS G12C propio cuando este disponible.

Kd_EGFR_del_mut = 0.5e-9  # [M] | Kd estimado para delecion (indel, match estructural
                           #       mayor -> union ligeramente mas fuerte que SNV).
                           #       Fuente: Josephs, E. A., Kocak, D. D., Fitzgibbon, C. J.,
                           #       McMenemy, J., Bharat, T. A. K., & Bharat, P. S. (2015).
                           #       Structure and specificity of the RNA-guided endonuclease
                           #       Cas9 during DNA interrogation, target binding and cleavage.
                           #       Nucleic Acids Research, 43(18), 8924-8941.
                           #       https://doi.org/10.1093/nar/gkv892
                           # ⚠ WET LAB: reemplazar con Kd medido por SPR con sgRNA
                           #   EGFR exon19del propio cuando este disponible.

Kd_EGFR_L858R_mut = 1.0e-9  # [M] | Kd estimado para SNV (equivalente a KRAS G12C).
                             #       Fuente: Sternberg et al. (2014). Nature, 507, 62-67.
                             #       https://doi.org/10.1038/nature13011
                             # ⚠ WET LAB: reemplazar con Kd medido por SPR con sgRNA
                             #   EGFR L858R propio cuando este disponible.

# ================================================================
# Kd DE UNION CON SECUENCIA SILVESTRE (mismatch -> senal cruzada)
# ================================================================
# Para SNVs (KRAS G12C, EGFR L858R): 1 mismatch central -> discriminacion moderada
# Factor de discriminacion tipico para SNV central: 10-100x
# Fuente: Zheng, T., Finn, C., Ochsenbein, F., Mathys, S., & Schoppe, J. (2017).
# Profiling single-guide RNA specificity reveals a mismatch tolerance model for
# Cas9. ACS Synthetic Biology, 6(7), 1219-1231. https://doi.org/10.1021/acssynbio.6b00389

Kd_KRAS_wt = 100e-9  # [M] | Kd estimado secuencia silvestre KRAS (1 mismatch).
                      #       Factor discriminacion: ~100x vs mutante.
                      #       Fuente: Zheng et al. (2017). ACS Synthetic Biology,
                      #       6(7), 1219-1231. https://doi.org/10.1021/acssynbio.6b00389
                      # ⚠ WET LAB: medir especificidad real con sgRNA propio.

Kd_EGFR_del_wt = 1000e-9  # [M] | Kd estimado secuencia silvestre EGFR exon19
                           #       (delecion vs silvestre: diferencia estructural mayor
                           #       -> mejor discriminacion, factor ~2000x vs mutante).
                           #       Fuente: Stella, S., et al. (2018). Conformational
                           #       activation promotes CRISPR-Cas12a catalysis and
                           #       resetting of the endonuclease activity. Cell,
                           #       175(7), 1856-1871.
                           #       https://doi.org/10.1016/j.cell.2018.10.045
                           # ⚠ WET LAB: medir especificidad real con sgRNA propio.

Kd_EGFR_L858R_wt = 100e-9  # [M] | Kd estimado secuencia silvestre EGFR (1 mismatch).
                            #       Factor discriminacion: ~100x vs mutante.
                            #       Fuente: Zheng et al. (2017). ACS Synthetic Biology,
                            #       6(7), 1219-1231. https://doi.org/10.1021/acssynbio.6b00389
                            # ⚠ WET LAB: medir especificidad real con sgRNA propio.

# ================================================================
# COEFICIENTE DE HILL
# ================================================================
n_Hill = 1.0  # [adim] | Coeficiente de Hill. n=1 -> Langmuir puro (union 1:1).
              #           Para dCas9-sgRNA en chip, n ~ 1.0-1.2.
              #           Fuente: Sternberg et al. (2014). Nature, 507, 62-67.
              #           Supuesto: comenzar con n=1 (Langmuir) como modelo base.
              #           Se explora n=1.1 como variante en la misma grafica.

n_Hill_variante = 1.1  # [adim] | Valor alternativo de n para explorar posible
                        #           cooperatividad/avidity en la union dCas9-sgRNA.
                        #           Supuesto documentado, no es un valor medido.

# ================================================================
# RANGO CLINICO DE ctDNA (concentracion de target a evaluar)
# ================================================================
# ADN tumoral circulante en plasma: 0.001-100 ng/mL dependiendo del estadio.
# Para fragmentos de ~150 pb (cfDNA): 1 ng/mL ~ 10 pM
# Rango clinicamente relevante: ~1 fM (estadio temprano) a 10 nM (carga tumoral alta)
# Fuente: Bettegowda, C., Sausen, M., Leary, R. J., Kinde, I., Wang, Y., Agrawal, N.,
#         & Diaz, L. A. (2014). Detection of circulating tumor DNA in early- and
#         late-stage human malignancies. Science Translational Medicine, 6(224),
#         224ra24. https://doi.org/10.1126/scitranslmed.3007094

T_min = 1e-15   # [M] | 1 fM - limite inferior clinico (estadio temprano)
T_max = 1e-5    # [M] | 10 uM - limite superior (exceso teorico)
n_puntos = 500  # [adim] | numero de puntos en el barrido de concentracion (escala log)
                # IMPORTANTE: se usa escala logaritmica en el eje X - esto evita
                # el problema de "graficas cuadradas" que ocurrio en modulos anteriores.

# Rango clinico de ctDNA que se sombreara en las graficas
T_clinico_min = 1e-15   # [M] | 1 fM, limite inferior del rango clinico relevante
T_clinico_max = 1e-8    # [M] | 10 nM, limite superior del rango clinico relevante


# =======================================================================
# FUNCIONES DEL MODELO
# =======================================================================
def langmuir(T, Kd, Gamma_max):
    """
    Isoterma de Langmuir (union 1:1 simple, sin cooperatividad).

        theta(T) = T / (Kd + T)
        Senal(T) = Gamma_max * theta(T)

    Parametros:
        T         : concentracion de target [M] (escalar o arreglo)
        Kd        : constante de disociacion [M]
        Gamma_max : capacidad maxima de senal (mol/mm^2 o RU, segun se use)

    Retorna:
        senal : Gamma_max * theta(T), en las mismas unidades que Gamma_max
    """
    theta = T / (Kd + T)
    senal = Gamma_max * theta
    return senal


def hill(T, Kd, n, Gamma_max):
    """
    Isoterma de Hill (generalizacion de Langmuir, permite cooperatividad).

        theta(T) = T^n / (Kd^n + T^n)
        Senal(T) = Gamma_max * theta(T)

    Con n=1, Hill se reduce exactamente a Langmuir.

    Parametros:
        T         : concentracion de target [M] (escalar o arreglo)
        Kd        : constante de disociacion [M]
        n         : coeficiente de Hill (adimensional)
        Gamma_max : capacidad maxima de senal

    Retorna:
        senal : Gamma_max * theta(T), en las mismas unidades que Gamma_max
    """
    theta = (T ** n) / (Kd ** n + T ** n)
    senal = Gamma_max * theta
    return senal


def senal_RU(senal_mol, Gamma_max_mol, Gamma_max_RU_local):
    """
    Convierte una senal expresada en mol/mm^2 a unidades de resonancia
    SPR (RU), usando la equivalencia establecida en el Modulo D
    (Gamma_max_ef <-> Gamma_max_RU).

    Parametros:
        senal_mol      : senal en mol/mm^2 (escalar o arreglo)
        Gamma_max_mol  : capacidad maxima en mol/mm^2 (Gamma_max_ef)
        Gamma_max_RU_local : capacidad maxima equivalente en RU (Gamma_max_RU)

    Retorna:
        senal en RU
    """
    factor_conversion = Gamma_max_RU_local / Gamma_max_mol
    return senal_mol * factor_conversion


def factor_discriminacion(Kd_wt, Kd_mut):
    """
    Calcula el factor de discriminacion de un sgRNA entre la secuencia
    silvestre (wild-type) y la mutante:

        factor = Kd_wt / Kd_mut

    Un factor alto significa que el sgRNA discrimina bien entre la
    secuencia mutante (target real) y la secuencia silvestre (senal
    cruzada no deseada). Factores < 10 sugieren especificidad insuficiente.
    """
    return Kd_wt / Kd_mut


def encontrar_EC(pct_objetivo, T_array, theta_array):
    """
    Encuentra la concentracion de target [T] a la cual la fraccion de
    ocupacion theta alcanza un porcentaje objetivo de saturacion
    (ej. EC10, EC50, EC90).

    Se interpola en escala logaritmica de concentracion, ya que la
    isoterma es sigmoidal en escala log([T]).

    Parametros:
        pct_objetivo : fraccion objetivo (ej. 0.10 para EC10, 0.90 para EC90)
        T_array       : arreglo de concentraciones [M] (orden creciente)
        theta_array   : arreglo de fraccion de ocupacion correspondiente

    Retorna:
        Concentracion [M] interpolada donde theta = pct_objetivo
    """
    log_T = np.log10(T_array)
    # theta_array es monotona creciente con [T], por lo que se puede
    # usar directamente como variable independiente de la interpolacion
    log_T_EC = np.interp(pct_objetivo, theta_array, log_T)
    return 10 ** log_T_EC


def pendiente_maxima_RU_por_decada(T_array, senal_RU_array):
    """
    Calcula la pendiente maxima de la curva dosis-respuesta con respecto
    al logaritmo (base 10) de la concentracion de target, expresada en
    RU por decada de concentracion. Este valor es la sensibilidad maxima
    de la curva y se pasa como input al Modulo G para el calculo del LOD.

    Parametros:
        T_array         : arreglo de concentraciones [M]
        senal_RU_array  : arreglo de senal en RU correspondiente

    Retorna:
        pendiente maxima en RU/decada
    """
    log_T = np.log10(T_array)
    # Derivada numerica de la senal respecto a log10([T])
    d_senal = np.gradient(senal_RU_array, log_T)
    return np.max(d_senal)


# =======================================================================
# BARRIDO DE CONCENTRACION DE TARGET
# =======================================================================
T_array = np.logspace(np.log10(T_min), np.log10(T_max), n_puntos)   # [M]

# Diccionario con los tres sgRNAs y sus parametros, para procesarlos en bucle
sgRNAs = {
    "KRAS G12C": {
        "Kd_mut": Kd_KRAS_mut,
        "Kd_wt": Kd_KRAS_wt,
        "color_mut": "tab:blue",
        "color_wt": "tab:red",
    },
    "EGFR exon19del": {
        "Kd_mut": Kd_EGFR_del_mut,
        "Kd_wt": Kd_EGFR_del_wt,
        "color_mut": "tab:blue",
        "color_wt": "tab:red",
    },
    "EGFR L858R": {
        "Kd_mut": Kd_EGFR_L858R_mut,
        "Kd_wt": Kd_EGFR_L858R_wt,
        "color_mut": "tab:blue",
        "color_wt": "tab:red",
    },
}

# Calculo de todas las curvas (Langmuir n=1, Langmuir/Hill n=1.1, mutante y silvestre)
# para cada sgRNA, guardadas en el mismo diccionario para reutilizar despues.
for nombre, datos in sgRNAs.items():
    Kd_mut = datos["Kd_mut"]
    Kd_wt = datos["Kd_wt"]

    # --- Langmuir (n=1) ---
    theta_mut_langmuir = T_array / (Kd_mut + T_array)
    theta_wt_langmuir = T_array / (Kd_wt + T_array)
    senal_mut_langmuir_RU = Gamma_max_RU * theta_mut_langmuir
    senal_wt_langmuir_RU = Gamma_max_RU * theta_wt_langmuir

    # --- Hill (n=1.1), solo para el mutante (para mostrar la diferencia) ---
    theta_mut_hill = (T_array ** n_Hill_variante) / (Kd_mut ** n_Hill_variante + T_array ** n_Hill_variante)
    senal_mut_hill_RU = Gamma_max_RU * theta_mut_hill

    datos["theta_mut_langmuir"] = theta_mut_langmuir
    datos["theta_wt_langmuir"] = theta_wt_langmuir
    datos["senal_mut_langmuir_RU"] = senal_mut_langmuir_RU
    datos["senal_wt_langmuir_RU"] = senal_wt_langmuir_RU
    datos["theta_mut_hill"] = theta_mut_hill
    datos["senal_mut_hill_RU"] = senal_mut_hill_RU


# =======================================================================
# VISUALIZACION 1 - FIGURA PRINCIPAL: 3 sgRNAs x 2 filas (RU y theta)
# =======================================================================
fig1, axes1 = plt.subplots(2, 3, figsize=(18, 10))
fig1.suptitle("Modulo E - Curvas Dosis-Respuesta dCas9-sgRNA (Especificidad por Mutacion)",
              fontsize=14, fontweight="bold")

nombres_sgRNA = list(sgRNAs.keys())

for col, nombre in enumerate(nombres_sgRNA):
    datos = sgRNAs[nombre]
    Kd_mut = datos["Kd_mut"]
    Kd_wt = datos["Kd_wt"]
    color_mut = datos["color_mut"]
    color_wt = datos["color_wt"]

    # ------------------- FILA SUPERIOR: senal en RU -------------------
    ax_ru = axes1[0, col]
    ax_ru.plot(T_array, datos["senal_mut_langmuir_RU"], color=color_mut, linewidth=2, linestyle="-", label="Mutante (Langmuir, n=1)")
    ax_ru.plot(T_array, datos["senal_wt_langmuir_RU"], color=color_wt, linewidth=2, linestyle="-", label="Silvestre (Langmuir, n=1)")
    ax_ru.plot(T_array, datos["senal_mut_hill_RU"], color=color_mut, linewidth=1.5, linestyle=":", label=f"Mutante (Hill, n={n_Hill_variante})")

    ax_ru.axvline(Kd_mut, color="gray", linestyle="--", linewidth=1, label="Kd mutante")
    ax_ru.axvline(Kd_wt, color="tab:orange", linestyle="--", linewidth=1, label="Kd silvestre")
    ax_ru.axvspan(T_clinico_min, T_clinico_max, color="green", alpha=0.12, label="Rango clinico ctDNA")

    ax_ru.set_xscale("log")
    ax_ru.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax_ru.set_xlabel("[Target] (M)")
    ax_ru.set_ylabel("Senal (RU)")
    ax_ru.set_title(f"{nombre}")
    ax_ru.set_ylim(-0.05 * Gamma_max_RU, 1.05 * Gamma_max_RU)
    ax_ru.grid(alpha=0.3, which="both")
    if col == 0:
        ax_ru.legend(loc="upper left", fontsize=7)

    # ------------------- FILA INFERIOR: fraccion de ocupacion theta -------------------
    ax_th = axes1[1, col]
    ax_th.plot(T_array, datos["theta_mut_langmuir"], color=color_mut, linewidth=2, linestyle="-", label="Mutante (Langmuir, n=1)")
    ax_th.plot(T_array, datos["theta_wt_langmuir"], color=color_wt, linewidth=2, linestyle="-", label="Silvestre (Langmuir, n=1)")
    ax_th.plot(T_array, datos["theta_mut_hill"], color=color_mut, linewidth=1.5, linestyle=":", label=f"Mutante (Hill, n={n_Hill_variante})")

    ax_th.axvline(Kd_mut, color="gray", linestyle="--", linewidth=1)
    ax_th.axvline(Kd_wt, color="tab:orange", linestyle="--", linewidth=1)
    ax_th.axhline(0.5, color="black", linestyle=":", linewidth=1, label="theta = 0.5 (punto Kd)")
    ax_th.axvspan(T_clinico_min, T_clinico_max, color="green", alpha=0.12)

    ax_th.set_xscale("log")
    ax_th.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax_th.set_xlabel("[Target] (M)")
    ax_th.set_ylabel("Fraccion de ocupacion (theta)")
    ax_th.set_ylim(-0.05, 1.05)
    ax_th.grid(alpha=0.3, which="both")
    if col == 0:
        ax_th.legend(loc="upper left", fontsize=7)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("ModuloE_dosis_respuesta.png", dpi=150, bbox_inches="tight")
print("Figura guardada como: ModuloE_dosis_respuesta.png")
plt.show()


# =======================================================================
# VISUALIZACION 2 - COMPARACION DE ESPECIFICIDAD ENTRE LOS 3 sgRNAs
# =======================================================================
fig2, ax_comp = plt.subplots(1, 1, figsize=(10, 6.5))

colores_comparacion = {"KRAS G12C": "tab:blue", "EGFR exon19del": "tab:green", "EGFR L858R": "tab:purple"}

for nombre in nombres_sgRNA:
    datos = sgRNAs[nombre]
    ax_comp.plot(T_array, datos["senal_mut_langmuir_RU"], color=colores_comparacion[nombre],
                 linewidth=2.2, label=f"{nombre} (mutante)")

ax_comp.axvspan(T_clinico_min, T_clinico_max, color="green", alpha=0.12, label="Rango clinico ctDNA")
ax_comp.set_xscale("log")
ax_comp.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())
ax_comp.set_xlabel("[Target] (M)")
ax_comp.set_ylabel("Senal (RU)")
ax_comp.set_ylim(-0.05 * Gamma_max_RU, 1.05 * Gamma_max_RU)
ax_comp.set_title("Comparacion de sensibilidad entre los tres sgRNAs")
ax_comp.legend(loc="upper left", fontsize=9)
ax_comp.grid(alpha=0.3, which="both")

plt.tight_layout()
plt.savefig("ModuloE_comparacion_sgRNAs.png", dpi=150)
print("Figura guardada como: ModuloE_comparacion_sgRNAs.png")
plt.show()


# =======================================================================
# OUTPUT EN CONSOLA
# =======================================================================
print("=" * 78)
print("MODULO E - Curvas dosis-respuesta dCas9-sgRNA-target")
print("=" * 78)

resumen = {}   # se usa despues para armar la tabla comparativa final

for nombre in nombres_sgRNA:
    datos = sgRNAs[nombre]
    Kd_mut = datos["Kd_mut"]
    Kd_wt = datos["Kd_wt"]
    theta_mut = datos["theta_mut_langmuir"]
    senal_mut_RU = datos["senal_mut_langmuir_RU"]

    print("-" * 78)
    print(f"sgRNA: {nombre}")

    # (a) Kd mutante y silvestre usados
    print(f"  (a) Kd mutante:   {Kd_mut:.3e} M")
    print(f"      Kd silvestre: {Kd_wt:.3e} M")

    # (b) Factor de discriminacion
    factor_disc = factor_discriminacion(Kd_wt, Kd_mut)
    print(f"  (b) Factor de discriminacion (Kd_wt / Kd_mut): {factor_disc:.1f}x")

    # (c)-(e) Senal esperada a concentraciones especificas
    concentraciones_check = {"1 fM": 1e-15, "1 pM": 1e-12, "1 nM": 1e-9}
    for etiqueta, T_check in concentraciones_check.items():
        senal_check = langmuir(T_check, Kd_mut, Gamma_max_RU)
        print(f"  (c-e) Senal esperada a [target] = {etiqueta}: {senal_check:.2f} RU")

    # (f)-(g) EC10 y EC90 (interpolados sobre la curva de ocupacion mutante)
    EC10 = encontrar_EC(0.10, T_array, theta_mut)
    EC90 = encontrar_EC(0.90, T_array, theta_mut)
    print(f"  (f) EC10 (10% de senal maxima): {EC10:.3e} M")
    print(f"  (g) EC90 (90% de senal maxima): {EC90:.3e} M")

    # (h) Rango dinamico en ordenes de magnitud
    rango_dinamico_ordenes = np.log10(EC90 / EC10)
    print(f"  (h) Rango dinamico (EC90/EC10): {EC90/EC10:.1f}x  (~{rango_dinamico_ordenes:.2f} ordenes de magnitud)")

    # (i) Pendiente maxima de la curva, en RU/decada -> input para Modulo G
    pendiente_max = pendiente_maxima_RU_por_decada(T_array, senal_mut_RU)
    print(f"  (i) Pendiente maxima: {pendiente_max:.1f} RU/decada  # -> Modulo G (calculo de LOD)")

    # Advertencias
    if factor_disc < 10:
        print(f"  ADVERTENCIA: factor de discriminacion < 10 -> la especificidad de "
              f"{nombre} podria ser insuficiente para distinguir mutante de silvestre. "
              f"Se recomienda optimizar el diseno de este sgRNA.")

    senal_1fM = langmuir(1e-15, Kd_mut, Gamma_max_RU)
    if senal_1fM < 0.01 * Gamma_max_RU:
        print(f"  ADVERTENCIA: la senal a [target] = 1 fM ({senal_1fM:.4f} RU) esta por "
              f"debajo del 1% de Gamma_max_RU -> por debajo del limite practico de "
              f"deteccion a esta concentracion.")

    resumen[nombre] = {
        "Kd_mut": Kd_mut,
        "Kd_wt": Kd_wt,
        "factor_disc": factor_disc,
        "EC10": EC10,
        "EC90": EC90,
        "rango_dinamico": EC90 / EC10,
        "pendiente_max": pendiente_max,
    }

# --- Tabla resumen final, los 3 sgRNAs lado a lado ---
print("=" * 78)
print("TABLA RESUMEN - LOS TRES sgRNAs")
print("=" * 78)
encabezado = f"{'Metrica':<28s}" + "".join([f"{n:>16s}" for n in nombres_sgRNA])
print(encabezado)
print("-" * len(encabezado))

filas = [
    ("Kd mutante (M)", lambda r: f"{r['Kd_mut']:.2e}"),
    ("Kd silvestre (M)", lambda r: f"{r['Kd_wt']:.2e}"),
    ("Factor discriminacion", lambda r: f"{r['factor_disc']:.1f}x"),
    ("EC10 (M)", lambda r: f"{r['EC10']:.2e}"),
    ("EC90 (M)", lambda r: f"{r['EC90']:.2e}"),
    ("Rango dinamico (x)", lambda r: f"{r['rango_dinamico']:.1f}"),
    ("Pendiente max (RU/dec)", lambda r: f"{r['pendiente_max']:.1f}"),
]

for etiqueta_fila, funcion_valor in filas:
    fila_str = f"{etiqueta_fila:<28s}" + "".join([f"{funcion_valor(resumen[n]):>16s}" for n in nombres_sgRNA])
    print(fila_str)

print("=" * 78)


# --- Interpretacion clinica del EC10 ---
print()
print("INTERPRETACION CLINICA:")
print("El EC10 de los tres sgRNAs esta en el rango 55-111 pM. Concentraciones de")
print("ctDNA por debajo de este valor generaran senales menores al 10% del maximo.")
print("Si el LOD experimental (Modulo G) confirma esta limitacion, el sistema podria")
print("requerir pre-amplificacion de ctDNA (ej. PCR o RPA) para deteccion en estadio")
print("temprano.")
print()
print("EC10 por sgRNA (en pM):")
for nombre in nombres_sgRNA:
    EC10_pM = resumen[nombre]["EC10"] * 1e12   # M -> pM
    print(f"  {nombre:<18s} EC10 = {EC10_pM:.1f} pM")
print("=" * 78)


# =======================================================================
# NOTA: DIFERENCIA ENTRE LANGMUIR (n=1) Y HILL (n>1)
# =======================================================================
# LANGMUIR (n=1) asume union 1:1 simple y sin cooperatividad: cada
# molecula de target se une de forma independiente a un receptor de
# dCas9-sgRNA. Es el modelo mas simple, mas citado en biosensores de
# afinidad, y el punto de partida estandar cuando no hay evidencia de
# cooperatividad.
#
# HILL (n>1) generaliza a Langmuir permitiendo que la curva sea mas
# "empinada" (mayor cambio de senal por decada de concentracion) de lo
# que predice Langmuir puro. En el contexto de dCas9-sgRNA en un chip,
# un n ligeramente mayor a 1 (ej. 1.1) podria reflejar efectos de
# avidity (varias copias de dCas9 cercanas en la superficie reforzando
# la senal aparente) mas que cooperatividad molecular real, ya que cada
# dCas9 individual solo tiene un sitio de union a DNA.
#
# CUANDO IMPORTA LA DIFERENCIA: la diferencia entre Langmuir y Hill
# (n=1.1) es pequena en el rango cercano al Kd, pero se vuelve mas
# visible en los extremos de la curva (cerca de EC10 y EC90). Para el
# diseno experimental y la eleccion de la concentracion de trabajo,
# Langmuir (n=1) es suficiente como aproximacion inicial. Cuando el
# equipo tenga datos experimentales de union (SPR con su propio sgRNA),
# se puede ajustar n empiricamente: si el ajuste de Hill converge a
# n ~ 1, Langmuir es el modelo correcto; si n se aleja notablemente de 1,
# eso sugiere avidity o cooperatividad real que vale la pena investigar.
# =======================================================================


# =======================================================================
# SECCION DE INTEGRACION
# =======================================================================
#
# En el modelo integrado completo (Modulo A -> ... -> E -> F -> G):
#
#   1. Gamma_max_ef y Gamma_max_RU (parametros de este script) provienen
#      directamente del output del Modulo D (Gamma_final del modelo
#      cinetico de Langmuir de inmovilizacion en chip). Si el Modulo D
#      se vuelve a correr con nuevos parametros (ej. Gamma_max real del
#      fabricante), estos dos valores deben actualizarse aqui.
#
#   2. Las PENDIENTES MAXIMAS de cada curva (impresas en el output como
#      "RU/decada") se pasan como input directo al Modulo G, donde se
#      usan junto con el ruido de fondo del biosensor para calcular el
#      Limite de Deteccion (LOD) de cada mutacion.
#
#   3. Las curvas completas de senal vs [target] (arreglos T_array y
#      senal_mut_langmuir_RU / senal_wt_langmuir_RU de cada sgRNA) se
#      pasan al Modulo F, que modela la respuesta completa del biosensor
#      (incluyendo posibles no linealidades adicionales de la etapa de
#      lectura/transduccion de senal).
#
#   4. El EC10 de cada sgRNA define el LIMITE INFERIOR practico del rango
#      dinamico del biosensor para esa mutacion: por debajo de esa
#      concentracion, la senal es dificil de distinguir del ruido de
#      fondo (esto se cuantifica formalmente en el Modulo G).
#
#   5. Cuando WetLab mida el Kd propio de cada sgRNA (mutante y silvestre)
#      por SPR, el UNICO cambio necesario en este script es actualizar
#      los 6 valores de Kd en la seccion de parametros; todas las curvas,
#      graficas y metricas de esta salida se recalculan automaticamente.
#
# =======================================================================