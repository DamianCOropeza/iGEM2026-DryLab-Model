#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=======================================================================
MODULO C - FRACCION FUNCIONAL PURIFICADA
=======================================================================
Proyecto iGEM 2026 - Deteccion de mutaciones oncogenicas en ctDNA
Modelado matematico de biologia sintetica (Dry Lab)

Este modulo integra aritmeticamente los outputs de los Modulos A y B para
calcular cuanta proteina dCas9-AviTag FUNCIONAL, biotinilada y PURIFICADA
queda disponible para presentar al chip de streptavidina (Modulo D).

Pipeline:
    Modulo A (5 ODEs)      -> P_dCas9(20h) con IC90%
    Modulo B (Michaelis-Menten) -> % biotinilacion ~ 100%
             |
             v  (ambos entran aqui)
    MODULO C: P_funcional = P_dCas9(20h) x phi_16 x pct_biotin x eta_pur
             |
             v
    Modulo D (Langmuir, inmovilizacion en chip)

# ============================================================
# INSTALACION DE DEPENDENCIAS (ejecutar en terminal una sola vez)
# ============================================================
# conda activate pythonProjectya
# conda install numpy scipy matplotlib
#
# Ejecutar:
# python modulo_C.py
# ============================================================

=======================================================================
CONTEXTO Y SUPUESTOS DEL MODULO
=======================================================================
1. Este modulo NO tiene ecuaciones diferenciales: es una integracion
   ARITMETICA de 4 factores multiplicativos, cada uno con su propia
   fuente y su propia incertidumbre. La complejidad esta en propagar
   esa incertidumbre de forma rigurosa (Monte Carlo), no en la dinamica
   temporal (eso ya se resolvio en los Modulos A y B).

2. pct_biotin ~ 100% (confirmado en el Modulo B): dado que [S_0] << Km_S
   (regimen de primer orden) y la biotina intracelular (5000 nM) esta
   en exceso masivo sobre S_0 en todos los escenarios evaluados, la
   biotinilacion se completa practicamente por entero dentro de las 24h
   de reaccion disponibles. Por eso pct_biotin aporta muy poca
   incertidumbre al resultado final, comparado con los otros factores.

3. phi_16 se aplica DESPUES de la purificacion (como factor conceptual
   sobre el pool purificado), no antes: la purificacion por afinidad
   His-tag/Ni-NTA no distingue entre proteina correctamente plegada y
   proteina mal plegada, ya que AMBAS conservan el His-tag accesible
   (a menos que el mal plegamiento oculte el tag, lo cual se ignora
   aqui como simplificacion). Por tanto eta_pur actua sobre la proteina
   TOTAL soluble, y phi_16 se aplica como la fraccion de ESE pool
   purificado que es funcionalmente activa.

4. TrxA (tiorredoxina, fusionada N-terminal) mejora la SOLUBILIDAD de
   la proteina (reduce la tendencia a formar cuerpos de inclusion), pero
   esto NO es lo mismo que garantizar el plegamiento CORRECTO del
   dominio dCas9: una proteina puede ser soluble y aun asi no plegar
   correctamente su sitio activo. Por eso phi_16 y eta_pur siguen
   siendo factores conceptualmente distintos en este modelo, aunque
   ambos esten relacionados con la "calidad" de la proteina producida.

5. Los resultados de este modulo son sensibles PRINCIPALMENTE a
   P_dCas9(20h) (que hereda la incertidumbre de k_R y f_rt del Modulo A)
   y a eta_pur (rendimiento de purificacion, todavia sin medir). El
   analisis de sensibilidad (tornado chart) mas abajo cuantifica esto.
=======================================================================
"""

# =======================================================================
# IMPORTS
# =======================================================================
import numpy as np                          # arreglos numericos, muestreo Monte Carlo (np.random.triangular)
import matplotlib.pyplot as plt              # generacion de graficas
import matplotlib.gridspec as gridspec       # layout de subplots 2x2
import scipy.stats as stats                  # utilidades estadisticas (percentiles, etc. via numpy tambien)

np.random.seed(42)   # reproducibilidad de las corridas Monte Carlo

# =======================================================================
# SECCION A: INPUTS DE MODULOS ANTERIORES
# =======================================================================

# --- Del Modulo A v2.0 ---
P_dCas9_20h_nominal = 20.145e-9  # [M] | ← Modulo A v2.0: P_dCas9(t=20h), corrida
                                  #        deterministica nominal, Radau, 16C.

P_dCas9_P5  = 5.78e-9    # [M] | ← Modulo A v2.0: percentil 5 del Monte Carlo (N=500)
P_dCas9_P50 = 17.43e-9   # [M] | ← Modulo A v2.0: mediana del Monte Carlo (N=500)
P_dCas9_P95 = 50.64e-9   # [M] | ← Modulo A v2.0: percentil 95 del Monte Carlo (N=500)

phi_16_nominal = 0.60    # [adim] | ← Modulo A v2.0.
                          #           Fuente: Vera, A., Gonzalez-Montalban, N., Aris, A.,
                          #           & Villaverde, A. (2007). The conformational quality
                          #           of insoluble recombinant proteins is enhanced at low
                          #           growth temperatures. Biotechnology and Bioengineering,
                          #           96(6), 1101-1106. https://doi.org/10.1002/bit.21218
phi_16_rel_unc = 0.15     # [adim] | ← Modulo A v2.0 (±15% de incertidumbre relativa)
                          # ⚠ WET LAB: confirmar por SDS-PAGE fraccion soluble/insoluble
                          #            a 16C post-lisis. PRIORIDAD ALTA.

# --- Del Modulo B ---
pct_biotin_nominal = 1.00   # [adim] | ← Modulo B: % biotinilacion ~ 100% en todos los
                             #           escenarios. [S_0] << Km_S (regimen primer orden)
                             #           y [biotina_intracelular] = 5000 nM >> S_0.
                             #           La biotina nunca es limitante en el sistema actual.
pct_biotin_rel_unc = 0.03   # [adim] | ← Modulo B: incertidumbre conservadora ±3%
                             #           (refleja posibles variaciones en [IPTG], T,
                             #           concentracion de BirA en diferentes escenarios)

# =======================================================================
# SECCION B: RENDIMIENTO DE PURIFICACION (parametro nuevo de este modulo)
# =======================================================================
eta_pur_nominal = 0.55   # [adim] | Rendimiento de purificacion His-tag Ni-NTA.
                          #          Valor central del rango de literatura para proteinas
                          #          grandes (>100 kDa) con fusion de solubilidad.
                          #          Rango tipico: 40-75% del lisado soluble.
                          #          Fuente: Bornhorst, J. A., & Falke, J. J. (2000).
                          #          Purification of proteins using polyhistidine affinity
                          #          tags. Methods in Enzymology, 326, 245-254.
                          #          https://doi.org/10.1016/S0076-6879(00)26058-8
                          # ⚠ WET LAB PRIORITARIO: medir por Bradford + SDS-PAGE/
                          #   densitometria en su primer lote de purificacion.
                          #   Calcular: (ug proteina en eluato) / (ug total en lisado x phi_16)
                          #   Este es el dato que mas diferencia hace entre un modelo
                          #   ilustrativo y un modelo predictivo de su proceso real.

eta_pur_min = 0.30       # [adim] | Limite inferior plausible (proteinas grandes, lisis
                          #          incompleta, perdidas en lavados).
                          #          Fuente: Lebendiker, M., & Danieli, T. (2014).
                          #          Production of prone-to-aggregate proteins.
                          #          FEBS Letters, 588(2), 236-246.
                          #          https://doi.org/10.1016/j.febslet.2013.10.044

eta_pur_max = 0.75       # [adim] | Limite superior plausible (condiciones optimizadas,
                          #          His-tag totalmente accesible gracias a TrxA como
                          #          tag de solubilidad).
                          #          Fuente: LaVallie, E. R., DiBlasio, E. A., Kovacic, S.,
                          #          Grant, K. L., Schendel, P. F., & McCoy, J. M. (1993).
                          #          A thioredoxin gene fusion expression system that
                          #          circumvents inclusion body formation in the E. coli
                          #          cytoplasm. Bio/Technology, 11(2), 187-193.
                          #          https://doi.org/10.1038/nbt0293-187
                          # Nota: TrxA (tiorredoxina) como tag de solubilidad N-terminal
                          # en la construccion 6xHis-TrxA-dCas9-AviTag mejora el
                          # rendimiento de purificacion respecto a construcciones sin el,
                          # particularmente para proteinas grandes como dCas9 (160 kDa).

eta_pur_rel_unc = 0.20   # [adim] | Incertidumbre relativa ±20% para distribucion
                          #          triangular en Monte Carlo. La moda es eta_pur_nominal.

# =======================================================================
# SECCION C: PESO MOLECULAR (para conversion de unidades)
# =======================================================================
# La construccion es: 6xHis-TrxA-dCas9(D10A/H840A)-AviTag
MW_dCas9_kDa = 160.0     # [kDa] | Peso molecular de dCas9 (SpCas9 con mutaciones
                          #         D10A y H840A). Mismo tamano que SpCas9 silvestre.
                          #         Fuente: Qi, L. S., Larson, M. H., Gilbert, L. A.,
                          #         Doudna, J. A., Weissman, J. S., Arkin, A. P., &
                          #         Lim, W. A. (2013). Repurposing CRISPR as an
                          #         RNA-guided platform for sequence-specific control of
                          #         gene expression. Cell, 152(5), 1173-1183.
                          #         https://doi.org/10.1016/j.cell.2013.02.022

MW_TrxA_kDa = 11.7       # [kDa] | Peso molecular de tiorredoxina A (TrxA) de E. coli.
                          #         Fuente: LaVallie et al. (1993). Bio/Technology, 11(2),
                          #         187-193. https://doi.org/10.1038/nbt0293-187

MW_AviTag_kDa = 1.2      # [kDa] | Peso molecular del AviTag (peptido de 15 aminoacidos:
                          #         GLNDIFEAQKIEWHE). ~80 Da x 15 aa ~ 1200 Da.
                          #         Fuente: Beckett, D., Kovaleva, E., & Schatz, P. J.
                          #         (1999). A minimal peptide substrate in biotin holoenzyme
                          #         synthetase-catalyzed biotinylation. Protein Science,
                          #         8(4), 921-929. https://doi.org/10.1110/ps.8.4.921

MW_6xHis_kDa = 0.9       # [kDa] | Peso molecular de 6xHistidina + linker (~900 Da).
                          #         Supuesto documentado: estimacion estandar para etiqueta
                          #         hexahistidina (6 x 137 Da ~ 820 Da + linker).

MW_total_kDa = MW_dCas9_kDa + MW_TrxA_kDa + MW_AviTag_kDa + MW_6xHis_kDa
# MW_total ~ 173.8 kDa -- se usa este valor en todas las conversiones de unidades

# =======================================================================
# SECCION D: PARAMETROS DE VOLUMEN (para conversion a masa)
# =======================================================================
V_cultivo_L = 1.0        # [L] | Volumen de cultivo de referencia para calcular
                          #       rendimiento en mg/L.
                          #       ⚠ WET LAB: ajustar al volumen real de su protocolo.
                          #       Decision de diseno experimental, no un dato bibliografico.

# NOTA: el Modulo A ya trabaja en concentraciones intracelulares (nM), por lo
# que la conversion directa nM -> ug/mL es valida para comparar con metodos
# como Bradford, sin necesidad de un factor adicional de volumen intracelular.


# =======================================================================
# FUNCION: CONVERSION nM -> ug/mL
# =======================================================================
def nM_a_ugmL(C_nM, MW_kDa):
    """
    Convierte una concentracion molar [nM] a concentracion masica [ug/mL],
    dado el peso molecular de la proteina [kDa].

    Derivacion paso a paso:
        C [nM] = C [mol/L] x 1e9
        MW [Da] = MW [kDa] x 1000
        C [g/L] = C [mol/L] x MW [g/mol]
        C [g/L] = (C_nM x 1e-9) x (MW_kDa x 1000)
        C [g/L] = C_nM x MW_kDa x 1e-6
        1 mg/L = 1 ug/mL (verificacion: 1 mg/L = 1000 ug / 1000 mL = 1 ug/mL)
        Por tanto: C [ug/mL] = C_nM x MW_kDa x 1e-3

    Parametros:
        C_nM   : concentracion en nM (escalar o arreglo)
        MW_kDa : peso molecular en kDa

    Retorna:
        Concentracion en ug/mL
    """
    return C_nM * MW_kDa * 1e-3


# =======================================================================
# CALCULO DETERMINISTICO (corrida nominal)
# =======================================================================
P_dCas9_20h_nominal_nM = P_dCas9_20h_nominal * 1e9   # convertir M -> nM para trabajar en nM

# Cascada de reduccion, paso a paso
paso_1_P_dCas9 = P_dCas9_20h_nominal_nM
paso_2_x_phi16 = paso_1_P_dCas9 * phi_16_nominal
paso_3_x_biotin = paso_2_x_phi16 * pct_biotin_nominal
paso_4_x_etapur = paso_3_x_biotin * eta_pur_nominal

P_funcional_nominal_nM = paso_4_x_etapur
P_funcional_nominal_M = P_funcional_nominal_nM * 1e-9
P_funcional_nominal_ugmL = nM_a_ugmL(P_funcional_nominal_nM, MW_total_kDa)

print("=" * 70)
print("MODULO C - Fraccion Funcional Purificada")
print("=" * 70)
print("\nINPUTS RECIBIDOS:")
print(f"  <- Modulo A: P_dCas9(20h) = {P_dCas9_20h_nominal_nM:.3f} nM "
      f"[IC90%: {P_dCas9_P5*1e9:.2f} - {P_dCas9_P95*1e9:.2f} nM]")
print(f"  <- Modulo A: phi_16 = {phi_16_nominal:.2f} (+-{phi_16_rel_unc*100:.0f}%)")
print(f"  <- Modulo B: pct_biotin = {pct_biotin_nominal*100:.1f}% (+-{pct_biotin_rel_unc*100:.0f}%)")
print(f"  Nuevo: eta_pur = {eta_pur_nominal:.2f} (rango literatura: {eta_pur_min:.2f} - "
      f"{eta_pur_max:.2f}) [WET LAB]")

print("\nFACTORES MULTIPLICATIVOS (corrida nominal):")
print(f"  P_dCas9(20h)          = {paso_1_P_dCas9:.3f} nM")
print(f"   x phi_16 ({phi_16_nominal:.2f})    -> {paso_2_x_phi16:.3f} nM")
print(f"   x pct_bio ({pct_biotin_nominal:.2f})   -> {paso_3_x_biotin:.3f} nM  "
      f"[biotinilacion no limitante]")
print(f"   x eta_pur ({eta_pur_nominal:.2f})   -> P_funcional = {paso_4_x_etapur:.3f} nM")

print(f"\nRESULTADO NOMINAL:")
print(f"  P_funcional = {P_funcional_nominal_nM:.3f} nM  -> Modulo D (inmovilizacion en chip)")
print(f"  P_funcional = {P_funcional_nominal_ugmL:.4f} ug/mL  (para verificar por Bradford)")
print(f"  (peso molecular total usado: {MW_total_kDa:.1f} kDa)")


# =======================================================================
# MONTE CARLO (N = 1000 simulaciones)
# =======================================================================
N_MC = 1000

# a) P_dCas9: distribucion triangular con moda=mediana de A, min=P5, max=P95
P_dCas9_muestras = np.random.triangular(P_dCas9_P5, P_dCas9_P50, P_dCas9_P95, size=N_MC) * 1e9  # a nM

# b) phi_16: triangular alrededor del nominal, +-15%
phi_16_lo = phi_16_nominal * (1 - phi_16_rel_unc)
phi_16_hi = phi_16_nominal * (1 + phi_16_rel_unc)
phi_16_muestras = np.random.triangular(phi_16_lo, phi_16_nominal, phi_16_hi, size=N_MC)

# c) pct_biotin: triangular acotada en [0.90, 1.00] (Modulo B confirma >97% en peor caso)
pct_biotin_lo = max(0.90, pct_biotin_nominal * (1 - pct_biotin_rel_unc))
pct_biotin_hi = min(1.00, pct_biotin_nominal * (1 + pct_biotin_rel_unc))
if pct_biotin_hi <= pct_biotin_lo:
    pct_biotin_hi = pct_biotin_lo + 1e-6  # proteccion numerica si el rango colapsa
pct_biotin_muestras = np.random.triangular(pct_biotin_lo, min(pct_biotin_nominal, pct_biotin_hi), pct_biotin_hi, size=N_MC)

# d) eta_pur: triangular con moda nominal, min/max de literatura
eta_pur_muestras = np.random.triangular(eta_pur_min, eta_pur_nominal, eta_pur_max, size=N_MC)

# e) Calculo de P_funcional para cada muestra
P_funcional_MC_nM = P_dCas9_muestras * phi_16_muestras * pct_biotin_muestras * eta_pur_muestras
P_funcional_MC_ugmL = nM_a_ugmL(P_funcional_MC_nM, MW_total_kDa)

# Estadisticos resumen
P_func_mediana = np.median(P_funcional_MC_nM)
P_func_p5 = np.percentile(P_funcional_MC_nM, 5)
P_func_p25 = np.percentile(P_funcional_MC_nM, 25)
P_func_p75 = np.percentile(P_funcional_MC_nM, 75)
P_func_p95 = np.percentile(P_funcional_MC_nM, 95)

P_func_mediana_ugmL = np.median(P_funcional_MC_ugmL)
P_func_p5_ugmL = np.percentile(P_funcional_MC_ugmL, 5)
P_func_p95_ugmL = np.percentile(P_funcional_MC_ugmL, 95)

frac_ge_1nM = np.mean(P_funcional_MC_nM >= 1.0) * 100.0
frac_ge_5nM = np.mean(P_funcional_MC_nM >= 5.0) * 100.0
frac_ge_10nM = np.mean(P_funcional_MC_nM >= 10.0) * 100.0

print(f"\nRESULTADO MONTE CARLO (N={N_MC}):")
print(f"  P_funcional: mediana = {P_func_mediana:.3f} nM  [IC90%: {P_func_p5:.3f} - {P_func_p95:.3f} nM]")
print(f"  P_funcional: mediana = {P_func_mediana_ugmL:.4f} ug/mL [IC90%: {P_func_p5_ugmL:.4f} - {P_func_p95_ugmL:.4f} ug/mL]")
print(f"  % corridas con P_func >= 1 nM : {frac_ge_1nM:.1f}%  (umbral minimo util para chip)")
print(f"  % corridas con P_func >= 5 nM : {frac_ge_5nM:.1f}%  (umbral chip bien cargado)")
print(f"  % corridas con P_func >= 10 nM: {frac_ge_10nM:.1f}%  (umbral optimo)")


# =======================================================================
# ANALISIS DE SENSIBILIDAD (tornado chart)
# =======================================================================
def montecarlo_variando_uno(variar, n=N_MC):
    """
    Corre Monte Carlo variando UNICAMENTE el factor indicado en 'variar'
    (uno de: 'P_dCas9', 'phi_16', 'pct_biotin', 'eta_pur'), manteniendo
    los demas fijos en su valor nominal/moda. Retorna el arreglo de
    P_funcional [nM] resultante.
    """
    P_dCas9_f = np.random.triangular(P_dCas9_P5, P_dCas9_P50, P_dCas9_P95, size=n) * 1e9 \
        if variar == "P_dCas9" else np.full(n, P_dCas9_P50 * 1e9)
    phi_16_f = np.random.triangular(phi_16_lo, phi_16_nominal, phi_16_hi, size=n) \
        if variar == "phi_16" else np.full(n, phi_16_nominal)
    pct_biotin_f = np.random.triangular(pct_biotin_lo, min(pct_biotin_nominal, pct_biotin_hi), pct_biotin_hi, size=n) \
        if variar == "pct_biotin" else np.full(n, pct_biotin_nominal)
    eta_pur_f = np.random.triangular(eta_pur_min, eta_pur_nominal, eta_pur_max, size=n) \
        if variar == "eta_pur" else np.full(n, eta_pur_nominal)

    return P_dCas9_f * phi_16_f * pct_biotin_f * eta_pur_f


factores_nombres = ["P_dCas9", "phi_16", "eta_pur", "pct_biotin"]
factores_etiquetas = {
    "P_dCas9": "P_dCas9 (heredado de Modulo A: k_R, f_rt)",
    "phi_16": "phi_16 (plegamiento, WetLab)",
    "eta_pur": "eta_pur (purificacion, WetLab)",
    "pct_biotin": "pct_biotin (Modulo B, ya casi cierto)",
}

std_por_factor = {}
for nombre in factores_nombres:
    resultado_f = montecarlo_variando_uno(nombre)
    std_por_factor[nombre] = np.std(resultado_f)

# Ordenar de mayor a menor contribucion a la varianza (std)
factores_ordenados = sorted(factores_nombres, key=lambda f: std_por_factor[f], reverse=True)
std_total = sum(std_por_factor.values())

print("\nFACTOR DOMINANTE EN INCERTIDUMBRE:")
for i, nombre in enumerate(factores_ordenados, start=1):
    pct_contribucion = std_por_factor[nombre] / std_total * 100.0
    print(f"  {i}. {factores_etiquetas[nombre]}: {pct_contribucion:.1f}% de la varianza total (std={std_por_factor[nombre]:.3f} nM)")

# --- Escenario "que pasa si mido eta_pur?" (valor del experimento) ---
P_dCas9_fijo_eta = np.random.triangular(P_dCas9_P5, P_dCas9_P50, P_dCas9_P95, size=N_MC) * 1e9
phi_16_fijo_eta = np.random.triangular(phi_16_lo, phi_16_nominal, phi_16_hi, size=N_MC)
pct_biotin_fijo_eta = np.random.triangular(pct_biotin_lo, min(pct_biotin_nominal, pct_biotin_hi), pct_biotin_hi, size=N_MC)
eta_pur_fijo = np.full(N_MC, eta_pur_nominal)   # SIN incertidumbre

P_funcional_sin_incert_etapur = P_dCas9_fijo_eta * phi_16_fijo_eta * pct_biotin_fijo_eta * eta_pur_fijo

IC90_ancho_completo = P_func_p95 - P_func_p5
IC90_sin_etapur_p5 = np.percentile(P_funcional_sin_incert_etapur, 5)
IC90_sin_etapur_p95 = np.percentile(P_funcional_sin_incert_etapur, 95)
IC90_ancho_sin_etapur = IC90_sin_etapur_p95 - IC90_sin_etapur_p5
reduccion_pct = (1 - IC90_ancho_sin_etapur / IC90_ancho_completo) * 100.0

print(f"\nEXPERIMENTO DE MAYOR IMPACTO (ROI experimental):")
print(f"  Si se mide eta_pur por Bradford, el IC90% se reduce de "
      f"[{P_func_p5:.3f} - {P_func_p95:.3f}] a [{IC90_sin_etapur_p5:.3f} - {IC90_sin_etapur_p95:.3f}] nM")
print(f"  Reduccion de incertidumbre: {reduccion_pct:.1f}% del ancho del intervalo")
print("=" * 70)


# =======================================================================
# VISUALIZACION - FIGURA 1: ANALISIS DE FACTORES (2x2)
# =======================================================================
fig1 = plt.figure(figsize=(16, 11))
gs1 = gridspec.GridSpec(2, 2, figure=fig1)
fig1.suptitle("Modulo C - Analisis de factores de la fraccion funcional purificada", fontsize=14, fontweight="bold")

ax_cascada = fig1.add_subplot(gs1[0, 0])
ax_tornado = fig1.add_subplot(gs1[0, 1])
ax_escenarios = fig1.add_subplot(gs1[1, 0])
ax_impacto = fig1.add_subplot(gs1[1, 1])

# --- Panel (1,1): Cascada de reduccion ---
etapas = ["P_dCas9\n(20h)", "x phi_16\n(x0.60)", "x pct_bio\n(x1.00)", "x eta_pur\n(x0.55)"]
valores_cascada = [paso_1_P_dCas9, paso_2_x_phi16, paso_3_x_biotin, paso_4_x_etapur]
colores_cascada = ["tab:green", "tab:orange", "tab:cyan", "tab:purple"]

barras_cascada = ax_cascada.bar(etapas, valores_cascada, color=colores_cascada, alpha=0.85, edgecolor="black")
for barra, valor in zip(barras_cascada, valores_cascada):
    ax_cascada.text(barra.get_x() + barra.get_width() / 2, valor, f"{valor:.2f} nM",
                     ha="center", va="bottom", fontsize=9)
ax_cascada.set_ylabel("Concentracion (nM)")
ax_cascada.set_title("Pipeline de reduccion: proteina producida -> funcional purificada")
ax_cascada.grid(alpha=0.3, axis="y")
ax_cascada.text(0.5, 0.02, "Nota: biotinilacion ~100% (no limitante) -> sin caida visible en ese paso",
                 transform=ax_cascada.transAxes, ha="center", fontsize=8, style="italic", color="gray")

# --- Panel (1,2): Tornado chart ---
nombres_tornado = [factores_etiquetas[f] for f in factores_ordenados]
valores_tornado = [std_por_factor[f] for f in factores_ordenados]
colores_tornado = ["tab:red", "tab:orange", "tab:blue", "tab:gray"]

ax_tornado.barh(nombres_tornado[::-1], valores_tornado[::-1], color=colores_tornado[::-1], edgecolor="black", alpha=0.85)
ax_tornado.set_xlabel("Desviacion estandar de P_funcional (nM)")
ax_tornado.set_title("Contribucion de cada factor a la incertidumbre de P_funcional")
ax_tornado.grid(alpha=0.3, axis="x")

# --- Panel (2,1): Escenarios (P5, Nominal, P95) ---
nombres_esc = ["Pesimista (P5)", "Nominal", "Optimista (P95)"]
valores_esc = [P_func_p5, P_funcional_nominal_nM, P_func_p95]
colores_esc = ["tab:red", "tab:blue", "tab:green"]

barras_esc = ax_escenarios.bar(nombres_esc, valores_esc, color=colores_esc, alpha=0.85, edgecolor="black")
for barra, valor in zip(barras_esc, valores_esc):
    ax_escenarios.text(barra.get_x() + barra.get_width() / 2, valor, f"{valor:.2f} nM",
                        ha="center", va="bottom", fontsize=9)
for umbral, etiqueta in [(1, "1 nM: minimo util"), (5, "5 nM: bien cargado"), (10, "10 nM: optimo")]:
    ax_escenarios.axhline(umbral, color="gray", linestyle="--", linewidth=1)
    ax_escenarios.text(2.6, umbral, f" {etiqueta}", va="bottom", fontsize=7.5, color="gray")
ax_escenarios.set_ylabel("P_funcional (nM)")
ax_escenarios.set_title("P_funcional en tres escenarios (heredados de Modulo A)")
ax_escenarios.grid(alpha=0.3, axis="y")

# --- Panel (2,2): Impacto de medir eta_pur ---
ax_impacto.hist(P_funcional_MC_nM, bins=30, color="tab:blue", alpha=0.5, label="Monte Carlo completo (toda incertidumbre)", density=True)
ax_impacto.hist(P_funcional_sin_incert_etapur, bins=30, color="tab:orange", alpha=0.5, label="eta_pur fijo (sin su incertidumbre)", density=True)
ax_impacto.set_xlabel("P_funcional (nM)")
ax_impacto.set_ylabel("Densidad de probabilidad")
ax_impacto.set_title("Impacto de medir eta_pur experimentalmente (valor del experimento)")
ax_impacto.legend(fontsize=8)
ax_impacto.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("ModuloC_analisis_factores.png", dpi=150)
print("\nFigura guardada como: ModuloC_analisis_factores.png")


# =======================================================================
# VISUALIZACION - FIGURA 2: HISTOGRAMA MONTE CARLO Y OUTPUT PARA MODULO D
# =======================================================================
fig2, (ax_hist_nM, ax_hist_ugmL) = plt.subplots(1, 2, figsize=(14, 6))
fig2.suptitle("Modulo C - Distribucion Monte Carlo de P_funcional (N=1000)", fontsize=14, fontweight="bold")

# --- Panel izquierdo: distribucion en nM ---
ax_hist_nM.hist(P_funcional_MC_nM, bins=35, color="tab:purple", alpha=0.75, edgecolor="black")
ax_hist_nM.axvline(P_func_p5, color="black", linestyle="--", linewidth=1.2, label=f"P5 = {P_func_p5:.2f} nM")
ax_hist_nM.axvline(P_func_mediana, color="black", linewidth=2, label=f"Mediana = {P_func_mediana:.2f} nM")
ax_hist_nM.axvline(P_func_p95, color="black", linestyle="--", linewidth=1.2, label=f"P95 = {P_func_p95:.2f} nM")
ax_hist_nM.axvspan(P_func_p5, P_func_p95, color="tab:purple", alpha=0.1)
ax_hist_nM.set_xlabel("P_funcional (nM)")
ax_hist_nM.set_ylabel("Frecuencia")
ax_hist_nM.set_title("Distribucion Monte Carlo de P_funcional (N=1000)")
ax_hist_nM.legend(fontsize=8)
ax_hist_nM.grid(alpha=0.3)
ax_hist_nM.text(0.97, 0.75, f"Mediana: {P_func_mediana:.2f} nM\nIC90%: [{P_func_p5:.2f}, {P_func_p95:.2f}] nM",
                 transform=ax_hist_nM.transAxes, ha="right", fontsize=9,
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

# --- Panel derecho: distribucion en ug/mL ---
ax_hist_ugmL.hist(P_funcional_MC_ugmL, bins=35, color="teal", alpha=0.75, edgecolor="black")
ax_hist_ugmL.axvline(P_func_p5_ugmL, color="black", linestyle="--", linewidth=1.2)
ax_hist_ugmL.axvline(P_func_mediana_ugmL, color="black", linewidth=2)
ax_hist_ugmL.axvline(P_func_p95_ugmL, color="black", linestyle="--", linewidth=1.2)
ax_hist_ugmL.axvspan(P_func_p5_ugmL, P_func_p95_ugmL, color="teal", alpha=0.1)
ax_hist_ugmL.set_xlabel("P_funcional (ug/mL)")
ax_hist_ugmL.set_ylabel("Frecuencia")
ax_hist_ugmL.set_title("P_funcional en ug/mL -> comparable con Bradford experimental")
ax_hist_ugmL.grid(alpha=0.3)
ax_hist_ugmL.text(0.97, 0.85, "Para verificar experimentalmente\npor Bradford + SDS-PAGE",
                    transform=ax_hist_ugmL.transAxes, ha="right", fontsize=8.5, style="italic",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("ModuloC_distribucion_output.png", dpi=150)
print("Figura guardada como: ModuloC_distribucion_output.png")

plt.show()


# =======================================================================
# OUTPUT EN CONSOLA FINAL
# =======================================================================
print("\n" + "=" * 70)
print("PARAMETROS PENDIENTES DE WET LAB (por prioridad de impacto)")
print("=" * 70)
print("  1. k_R_ref   (-> actualizar en Modulo A)  -- impacto: CRITICO")
print("  2. f_rt      (-> actualizar en Modulo A)  -- impacto: CRITICO")
print("  3. eta_pur   (purificacion, este modulo)  -- impacto: ALTO")
print("  4. phi_16    (-> actualizar en Modulo A)  -- impacto: MEDIO")
print("=" * 70)
print("OUTPUT PARA MODULO D:")
print(f"  P_funcional_para_Langmuir = {P_funcional_nominal_nM:.3f} nM  [nominal]")
print(f"  P_funcional_IC90          = [{P_func_p5:.3f}, {P_func_p95:.3f}] nM  "
      f"[para analisis de sensibilidad en Modulo D]")
print("=" * 70)


# =======================================================================
# SECCION DE INTEGRACION CON MODULOS
# =======================================================================
#
#   1. P_funcional [nM] (calculado arriba, tanto el valor nominal como el
#      IC90% del Monte Carlo) entra al Modulo D como la concentracion de
#      proteina disponible para inmovilizacion (el parametro C_proteina /
#      c_dCas9 de la isoterma y cinetica de Langmuir). Se recomienda que
#      el Modulo D corra sus 3 escenarios (pesimista/nominal/optimista)
#      usando los P5/mediana/P95 de ESTE modulo, en vez de los que traia
#      heredados directamente de Modulo B/A (este modulo ya incorpora la
#      incertidumbre adicional de phi_16, pct_biotin y eta_pur, que antes
#      no estaba propagada hacia D).
#
#   2. Cuando WetLab mida eta_pur real (Bradford + SDS-PAGE del primer
#      lote de purificacion), el UNICO cambio necesario es actualizar
#      eta_pur_nominal (y opcionalmente eta_pur_min/max si el rango
#      medido no coincide con el de literatura) en este script.
#
#   3. Cuando el Modulo A actualice k_R_ref o f_rt (tras mediciones de
#      WetLab), actualizar P_dCas9_P5/P50/P95 en la Seccion A de este
#      script con los nuevos percentiles del Monte Carlo de Modulo A.
#
#   4. pct_biotin puede actualizarse si WetLab hace un ensayo de
#      biotinilacion cuantitativo (streptavidina-HRP + densitometria),
#      reemplazando el valor teorico (~100%, derivado del regimen
#      cinetico) por una medicion directa del pool de proteina real.
#
# =======================================================================