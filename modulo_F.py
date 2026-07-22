#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=======================================================================
MODULO F - SENAL SPR EN MODO DE INTERROGACION DE FASE
=======================================================================
Proyecto iGEM 2025 - Deteccion de mutaciones oncogenicas en ctDNA
Modelado matematico de biologia sintetica (Dry Lab)

Este modulo convierte la senal FISICA de union (Gamma, densidad superficial
de ctDNA unido al chip, calculada en el Modulo E) en la senal INSTRUMENTAL
que realmente mide un biosensor SPR operando en modo de interrogacion de
fase (Delta_phi, en grados). Es el puente entre la biofisica de union y el
calculo del limite de deteccion (LOD) que hara el Modulo G.

Pipeline:
    Modulo D -> Gamma_max_ef = 5.0e-14 mol/mm^2 (capacidad maxima del chip)
    Modulo E -> Gamma_ctDNA(c) = Gamma_max x c/(Kd+c)  [curvas dosis-respuesta]
                |
                v
    MODULO F: Gamma_ctDNA [mol/mm^2] -> Delta_phi [grados] (senal medible)
                |
                v
    Modulo G <- Delta_phi(c) y sigma_blank -> calculo de LOD

# ============================================================
# INSTALACION DE DEPENDENCIAS (ejecutar en terminal una sola vez)
# ============================================================
# conda activate iGEM
# conda install numpy scipy matplotlib
# pip install numpy scipy matplotlib   (alternativa sin conda)
#
# Verificar:
# python -c "import numpy, scipy, matplotlib; print('Dependencias OK')"
#
# Ejecutar:
# python modulo_F.py
# ============================================================

=======================================================================
CONTEXTO FISICO Y SUPUESTOS
=======================================================================
a) SPR de interrogacion de FASE vs. INTENSIDAD: los biosensores SPR
   "clasicos" (ej. Biacore) miden el ANGULO de resonancia (o la intensidad
   reflejada a angulo fijo). La interrogacion de FASE, en cambio, mide el
   cambio de FASE de la luz reflejada en el punto de resonancia. Cerca de
   la resonancia, la fase cambia muy abruptamente (una "singularidad de
   fase"), lo que da sensibilidades varios ordenes de magnitud mayores que
   los metodos de intensidad/angulo -- por eso se elige este modo para un
   biosensor que necesita detectar ctDNA a concentraciones muy bajas.

b) POR QUE S_phase (sensibilidad de fase) es mayor que en intensidad: la
   curva de fase vs. angulo de incidencia tiene una pendiente muy alta
   (casi un salto tipo Heaviside) exactamente en el angulo de resonancia,
   mientras que la curva de intensidad vs. angulo es mas suave. Esto se
   traduce en mas grados de cambio de fase por cada unidad de cambio de
   indice de refraccion (deg/RIU), en vez de mas % de cambio de reflectividad.

c) LA CADENA DE TRANSDUCCION, PASO A PASO:
     Gamma [mol/mm^2]      -> cuantas moleculas de ctDNA hay por area de chip
       x MW_ctDNA           -> Gamma_masa [g/mm^2], masa por area
       -> Delta_n [RIU]      -> cambio de indice de refraccion local, causado
                                 por el aumento de masa cerca de la superficie
       x S_phase             -> Delta_phi [deg], la senal que el instrumento
                                 realmente reporta

d) POR QUE dn/dc DE DNA != dn/dc DE PROTEINA: el incremento especifico del
   indice de refraccion (dn/dc) depende de la polarizabilidad por unidad de
   masa de la molecula. Los acidos nucleicos (mas cargados, mas hidratados)
   tienen un dn/dc ligeramente menor (~0.168 cm^3/g) que las proteinas
   (~0.185 cm^3/g). Como el ANALITO que se une en esta etapa es ctDNA (no
   mas proteina), se debe usar dn_dc_DNA, no dn_dc_protein, en el calculo
   de Delta_n causado por la union del target.

e) LA APROXIMACION "1 RU ~ 1e-6 RIU": es una convencion estandar de la
   industria (especialmente equipos tipo Biacore), derivada asumiendo
   dn/dc~0.185 cm^3/g (proteina) y d_eff~200 nm. Es una aproximacion, no
   una ley fisica exacta -- por eso este modulo implementa TAMBIEN la Ruta
   B (mas fundamental, via dn/dc explicito) y compara ambas.

f) SUPUESTO DE MODELO LINEAL: Delta_phi = K x Gamma es valido solo mientras
   Delta_phi se mantenga muy por debajo de 360 grados (idealmente <180,
   mejor aun <90) -- de lo contrario ocurre "phase wrapping" (la fase da la
   vuelta completa y ya no se puede distinguir de una fase menor) y el
   modelo lineal deja de ser valido. Este script verifica automaticamente
   esta condicion e imprime una advertencia si se viola.

g) SUPUESTO DE LANGMUIR 1:1: se hereda directamente del Modulo E. La
   ocupacion de receptores (y por tanto Gamma) sigue Gamma(c) =
   Gamma_max x c/(Kd+c), sin cooperatividad.

h) S_phase Y sigma_blank SON PLACEHOLDERS: ambos son especificos del
   instrumento SPR real que use el equipo (geometria optica, longitud de
   onda, metal del chip, condiciones de medicion). Los valores usados aqui
   vienen de la literatura de sistemas SPR DISTINTOS, solo como punto de
   partida razonable de orden de magnitud. CUALQUIER LOD calculado en este
   script es PRELIMINAR hasta que WetLab mida estos dos parametros con su
   propio instrumento.
=======================================================================
"""

# =======================================================================
# IMPORTS
# =======================================================================
import numpy as np                            # arreglos numericos
import matplotlib.pyplot as plt                # graficas
import matplotlib.gridspec as gridspec         # layouts de subplots complejos
import matplotlib.ticker as ticker             # formato de ejes log
from scipy.optimize import brentq              # busqueda numerica robusta de c_LOD

# =======================================================================
# SECCION A: INPUTS DE MODULOS ANTERIORES
# =======================================================================

# --- Del Modulo D (inmovilizacion en chip) ---
Gamma_max_ef = 5.0e-14   # [mol/mm^2] | <- Modulo D: densidad superficial maxima efectiva
                          #               de dCas9-AviTag inmovilizado en chip streptavidina.
                          #               Equivalente a 8000 RU para MW_dCas9 ~ 160 kDa.

# --- Del Modulo E (curvas dosis-respuesta) -- Kd para 3 sgRNAs ---
# KRAS G12C (mutacion puntual G>T en codon 12)
Kd_KRAS_mut = 1.0e-9     # [M] | <- Modulo E. Kd de dCas9-sgRNA_KRAS para ctDNA mutante.
                          #        Discriminacion ~100x respecto a wildtype.
Kd_KRAS_wt = 100.0e-9    # [M] | <- Modulo E. Kd de dCas9-sgRNA_KRAS para ctDNA wildtype.

# EGFR exon19del (delecion in-frame en exon 19, la mas frecuente: delE746-A750)
Kd_EGFR19_mut = 0.5e-9   # [M] | <- Modulo E. Kd para ctDNA mutante (delecion).
                          #        Discriminacion ~2000x respecto a wildtype.
Kd_EGFR19_wt = 1000.0e-9 # [M] | <- Modulo E. Kd para ctDNA wildtype.

# EGFR L858R (mutacion puntual T>G en codon 858)
Kd_EGFR858_mut = 1.0e-9  # [M] | <- Modulo E. Kd para ctDNA mutante.
                          #        Discriminacion ~100x respecto a wildtype.
Kd_EGFR858_wt = 100.0e-9 # [M] | <- Modulo E. Kd para ctDNA wildtype.

# =======================================================================
# SECCION B: PROPIEDADES OPTICAS -- VALORES DE LITERATURA
# =======================================================================
dn_dc_DNA_cm3g = 0.168    # [cm^3/g] | Incremento especifico del indice de refraccion
                          #             de acidos nucleicos (DNA) en solucion acuosa, EN LAS
                          #             UNIDADES ORIGINALES reportadas en literatura (cm^3/g).
                          #             Valor estandar para DNA en buffer fisiologico.
                          #             Fuente: Stengel, G., Zahn, R., & Hook, F. (2007).
                          #             DNA-induced programmable fusion of phospholipid
                          #             vesicles. Journal of the American Chemical Society,
                          #             129(31), 9584-9585. https://doi.org/10.1021/ja073200k
                          #             Nota: dn/dc para proteinas ~ 0.185 cm^3/g (ligeramente
                          #             mayor). El ctDNA (analito) usa 0.168 cm^3/g.

# CORRECCION DE UNIDADES: el resto del modulo (d_eff, Gamma_mol, Gamma_masa)
# trabaja consistentemente en MILIMETROS (d_eff esta en mm, Gamma_mol en
# mol/mm^2). dn_dc_DNA_cm3g esta en cm^3/g, una unidad de LONGITUD distinta
# (cm en vez de mm) escondida dentro de "cm^3". Mezclar cm^3/g con mm en la
# misma formula (K_B = S_phase x dn_dc_DNA x MW / d_eff) sin convertir
# introduce un error de escala. La conversion correcta es:
#     1 cm^3 = 1000 mm^3  ->  dn_dc_DNA_mm3g = dn_dc_DNA_cm3g x 1000
# Con esto, dn_dc_DNA queda en mm^3/g, consistente con d_eff en mm y
# Gamma_mol en mol/mm^2, y K_B resulta correctamente en [deg*mm^2/mol].
dn_dc_DNA = dn_dc_DNA_cm3g * 1000   # [mm^3/g] = 168 mm^3/g -- valor ya convertido,
                                     #             usado en todos los calculos de abajo

dn_dc_protein = 0.185     # [cm^3/g] | Incremento especifico del indice de refraccion
                          #             de proteinas en solucion acuosa. Usado solo para
                          #             contexto/comparacion (la capa de dCas9 inmovilizado
                          #             ya esta fija; el analito que cambia Delta_n aqui es
                          #             el ctDNA, no la proteina).
                          #             Fuente: De Feijter, J. A., Benjamins, J., &
                          #             Veer, F. A. (1978). Ellipsometry as a tool to
                          #             study the adsorption behavior of synthetic and
                          #             biopolymers at the air-water interface.
                          #             Biopolymers, 17(7), 1759-1772.
                          #             https://doi.org/10.1002/bip.1978.360170711

d_eff = 200.0e-6          # [mm] | Profundidad efectiva del campo evanescente del
                          #         plasmon de superficie (200 nm = 200e-6 mm).
                          #         Esta es la distancia caracteristica de decaimiento
                          #         del campo EM desde la superficie del chip.
                          #         Rango tipico en SPR visible/NIR: 150-300 nm.
                          #         Fuente: Homola, J. (2008). Surface plasmon resonance
                          #         sensors for detection of chemical and biological
                          #         species. Chemical Reviews, 108(2), 462-493.
                          #         https://doi.org/10.1021/cr068107d
                          # WET LAB: el valor exacto depende de la longitud de onda
                          #   del laser y el metal del chip (Au tipico). Puede calcularse
                          #   con las constantes dielectricas del sistema real.

# =======================================================================
# SECCION C: PESO MOLECULAR DEL ctDNA ANALITO
# =======================================================================
MW_ctDNA = 1.1e5          # [g/mol] | Peso molecular de fragmentos de ctDNA circulante.
                          #            El ctDNA en plasma corresponde principalmente a
                          #            fragmentos mononucleosomales (~167 pb).
                          #            MW(167 pb dsDNA) = 167 x 615.4 g/mol/pb ~ 102,772 Da
                          #            Se usa 1.1e5 Da como aproximacion redondeada.
                          #            Fuente: Snyder, M. W., Kircher, M., Hill, A. J.,
                          #            Daza, R. M., & Shendure, J. (2016). Cell-free DNA
                          #            comprises an in vivo nucleosome footprint that
                          #            informs its tissues-of-origin. Cell, 164(1-2),
                          #            57-68. https://doi.org/10.1016/j.cell.2015.11.050
                          # WET LAB: verificar el tamano real de los fragmentos de
                          #   ctDNA en las muestras clinicas de interes por
                          #   electroforesis en gel o analisis de fragmentos
                          #   (BioAnalyzer o TapeStation).

MW_pb = 615.4             # [g/mol/pb] | Peso molecular promedio por par de bases en
                          #               DNA doble cadena (secuencia aleatoria).
                          #               Calculado como promedio de: dAMP (331.2),
                          #               dTMP (322.2), dGMP (347.2), dCMP (307.2)
                          #               x 2 (doble cadena) / 4 = 615.4 g/mol/pb.
                          #               Fuente: Voet, D., & Voet, J. G. (2011).
                          #               Biochemistry (4th ed.). Wiley.

# =======================================================================
# SECCION D: PARAMETROS DEL INSTRUMENTO SPR (PLACEHOLDERS WET LAB)
# =======================================================================
S_phase = 1.0e4           # [deg/RIU] | Sensibilidad de fase del instrumento SPR.
                          #              Cuantos grados de cambio de fase corresponden
                          #              a un cambio de indice de refraccion de 1 RIU.
                          # WET LAB CRITICO: este valor es especifico de su sistema
                          #   SPR (geometria optica, longitud de onda, metal del chip).
                          #   NO existe un valor universal que aplique a su instrumento.
                          #   PLACEHOLDER basado en:
                          #   Grigorenko, A. N., Nikitin, P. I., & Kabashin, A. V.
                          #   (1999). Phase jumps and interferometric surface plasmon
                          #   resonance imaging. Applied Physics Letters, 75(25),
                          #   3917-3919. https://doi.org/10.1063/1.125493
                          #   Rango tipico en literatura: 10^3-10^5 deg/RIU (algunos
                          #   sistemas de metamateriales reportan hasta 5x10^4 deg/RIU).
                          #   Para obtener el valor real: calibrar con soluciones de
                          #   glicerol o sacarosa de concentracion conocida y Delta_n
                          #   calculado por tabla de indices de refraccion.

sigma_blank = 1.0e-4      # [deg] | Ruido de fondo del sistema SPR (desviacion estandar
                          #          de la senal de fase en mediciones de blanco = buffer
                          #          sin analito).
                          # WET LAB CRITICO: este valor es especifico de su instrumento,
                          #   configuracion optica y condiciones de medicion.
                          #   PLACEHOLDER basado en:
                          #   Nikitin, P. I., Beloglazov, A. A., Kochergin, V. E.,
                          #   Valeiko, M. V., & Ksenevich, T. I. (1999). Surface plasmon
                          #   resonance interferometry for biological and chemical sensing.
                          #   Sensors and Actuators B: Chemical, 54(1-2), 43-50.
                          #   https://doi.org/10.1016/S0925-4005(98)00310-1
                          #   Para obtener el valor real: correr >=20 mediciones de
                          #   buffer solo (sin analito) y calcular std(senal_fase).
                          #   PRIORIDAD ALTA: este es el parametro mas importante
                          #   para calcular el LOD real del biosensor.

# =======================================================================
# SECCION E: RANGO CLINICO DE ctDNA
# =======================================================================
c_min = 1.0e-15           # [M] | Concentracion minima clinicamente relevante de ctDNA.
                          #        1 fM = limite inferior reportado en pacientes
                          #        con cancer en estadios tempranos.
                          #        Fuente: Bettegowda, C., et al. (2014). Detection of
                          #        circulating tumor DNA in early- and late-stage human
                          #        malignancies. Science Translational Medicine, 6(224),
                          #        224ra24. https://doi.org/10.1126/scitranslmed.3007094

c_max = 10.0e-9           # [M] | Concentracion maxima clinicamente relevante de ctDNA.
                          #        10 nM en estadios avanzados.
                          #        Fuente: Bettegowda et al. (2014). Ibid.


# =======================================================================
# SECCION 5: CALCULO DE K_transduction Y CROSS-VALIDACION
# =======================================================================

# --- Ruta A: via conversion RU (conecta con Modulo D/E) ---
# 1 RU = 1 pg/mm^2 -> Gamma_RU = Gamma_mol x MW x 1e12
# 1 RU ~ 1e-6 RIU (conversion estandar Biacore, calibrada para dn/dc=0.185, d_eff=200nm)
def delta_phi_ruta_A(Gamma_mol):
    """Calcula Delta_phi [deg] a partir de Gamma [mol/mm^2] via la Ruta A (conversion RU)."""
    Gamma_RU = Gamma_mol * MW_ctDNA * 1e12       # [pg/mm^2] = [RU]
    Delta_n = Gamma_RU * 1e-6                     # [RIU], convencion estandar Biacore
    return S_phase * Delta_n                      # [deg]

K_transduction_A = S_phase * MW_ctDNA * 1e12 * 1e-6   # [deg*mm^2/mol], forma cerrada de Ruta A

# --- Ruta B: via dn/dc explicito (mas fundamental) ---
# K_B = S_phase x dn_dc_DNA x MW_ctDNA / d_eff   [deg*mm^2/mol]
# NOTA: dn_dc_DNA aqui ya esta en mm^3/g (convertido arriba desde cm^3/g),
# consistente con d_eff en mm y Gamma_mol en mol/mm^2.
K_transduction_B = S_phase * dn_dc_DNA * MW_ctDNA / d_eff

def delta_phi_ruta_B(Gamma_mol):
    """Calcula Delta_phi [deg] a partir de Gamma [mol/mm^2] via la Ruta B (dn/dc explicito)."""
    return K_transduction_B * Gamma_mol

# --- Cross-validacion ---
Delta_phi_max_A = delta_phi_ruta_A(Gamma_max_ef)
Delta_phi_max_B = delta_phi_ruta_B(Gamma_max_ef)
diferencia_rutas_pct = abs(K_transduction_A - K_transduction_B) / K_transduction_A * 100.0

# Se usa K_B (Ruta B) como el valor PRINCIPAL de aqui en adelante, tal como
# especifica el diseno del modulo (mas fundamental, no depende de la
# convencion empirica "1 RU ~ 1e-6 RIU" calibrada para OTRO tipo de analito).
K_transduction = K_transduction_B
Delta_phi_max = Delta_phi_max_B

print("=" * 70)
print("MODULO F - Senal SPR en Modo de Fase")
print("=" * 70)
print("FUNCION DE TRANSDUCCION:")
print(f"  dn/dc DNA               = {dn_dc_DNA_cm3g} cm^3/g = {dn_dc_DNA} mm^3/g  [Stengel et al. 2007]")
print(f"  d_eff (evanescente)      = {d_eff*1e6:.0f} nm      [Homola 2008, placeholder]")
print(f"  MW_ctDNA (~167 pb)       = {MW_ctDNA:,.0f} Da   [Snyder et al. 2016]")
print(f"  K_transduction (Ruta B)  = {K_transduction_B:.3e} deg*mm^2/mol")
print(f"  K_transduction (Ruta A)  = {K_transduction_A:.3e} deg*mm^2/mol")
print(f"  Diferencia A vs B        = {diferencia_rutas_pct:.1f}%  (aceptable si < 20%)")

if diferencia_rutas_pct > 20.0:
    print("  ADVERTENCIA: las rutas A y B difieren en mas de 20%. Esto refleja una")
    print("  inconsistencia de unidades entre la convencion empirica '1 RU~1e-6 RIU'")
    print("  (calibrada originalmente para dn/dc de PROTEINA ~0.185 cm^3/g y d_eff en")
    print("  escala de Biacore) y el calculo explicito via dn/dc del DNA con d_eff en")
    print("  mm (Ruta B). Se usa Ruta B como principal por ser mas fundamental, pero")
    print("  esta discrepancia debe resolverse con calibracion propia del instrumento")
    print("  (WetLab) antes de confiar en el LOD final del Modulo G.")

print(f"\n  Delta_phi_max (chip saturado, Ruta B) = {Delta_phi_max:.4e} deg")
if Delta_phi_max > 180.0:
    print("  Phase wrapping:  ADVERTENCIA -- Delta_phi_max > 180 deg, el modelo lineal")
    print("  pierde validez cerca de saturacion del chip.")
else:
    print("  Phase wrapping:  OK (Delta_phi_max < 180 deg, modelo lineal valido)")

print(f"\nPARAMETROS DEL INSTRUMENTO (PLACEHOLDERS):")
print(f"  S_phase (placeholder)    = {S_phase:.1e} deg/RIU  [Grigorenko et al. 1999]")
print(f"  sigma_blank (placeholder)= {sigma_blank:.1e} deg      [Nikitin et al. 1999]")
LOD_phi = 3.0 * sigma_blank
print(f"  LOD_phi = 3 x sigma_blank = {LOD_phi:.1e} deg")


# =======================================================================
# SECCION 6: CURVAS DOSIS-RESPUESTA EN SENAL DE FASE
# =======================================================================
c_array = np.logspace(-15, -7, 500)   # [M] de 1 fM a 100 nM

def gamma_langmuir(c, Kd):
    """Isoterma de Langmuir 1:1, heredada del Modulo E."""
    return Gamma_max_ef * c / (Kd + c)

def delta_phi_de_c(c, Kd):
    """Delta_phi(c) [deg] usando K_transduction (Ruta B) y la isoterma de Langmuir."""
    return K_transduction * gamma_langmuir(c, Kd)

# Curvas para los 3 sgRNAs, mutante y silvestre
phi_KRAS_mut = delta_phi_de_c(c_array, Kd_KRAS_mut)
phi_KRAS_wt = delta_phi_de_c(c_array, Kd_KRAS_wt)
phi_EGFR19_mut = delta_phi_de_c(c_array, Kd_EGFR19_mut)
phi_EGFR19_wt = delta_phi_de_c(c_array, Kd_EGFR19_wt)
phi_EGFR858_mut = delta_phi_de_c(c_array, Kd_EGFR858_mut)
phi_EGFR858_wt = delta_phi_de_c(c_array, Kd_EGFR858_wt)

sgRNAs_info = {
    "KRAS G12C": {"Kd_mut": Kd_KRAS_mut, "Kd_wt": Kd_KRAS_wt, "phi_mut": phi_KRAS_mut, "phi_wt": phi_KRAS_wt},
    "EGFR exon19del": {"Kd_mut": Kd_EGFR19_mut, "Kd_wt": Kd_EGFR19_wt, "phi_mut": phi_EGFR19_mut, "phi_wt": phi_EGFR19_wt},
    "EGFR L858R": {"Kd_mut": Kd_EGFR858_mut, "Kd_wt": Kd_EGFR858_wt, "phi_mut": phi_EGFR858_mut, "phi_wt": phi_EGFR858_wt},
}


# =======================================================================
# SECCION 7: CALCULO DE LOD (criterio 3-sigma)
# =======================================================================
def encontrar_c_LOD(Kd, LOD_phi_objetivo, c_lo=1e-18, c_hi=1e-5):
    """
    Busca numericamente (brentq) la concentracion c_LOD tal que
    Delta_phi(c_LOD) = LOD_phi_objetivo.

    CORRECCION: la busqueda se hace en ESCALA LOGARITMICA (sobre log10(c)),
    no directamente sobre c. Motivo: c abarca un rango de 13+ ordenes de
    magnitud (1e-18 a 1e-5 M), y la tolerancia absoluta por defecto de
    brentq (xtol~2e-12) es much mas grande que las concentraciones de
    interes (~1e-15 M). Buscando directamente en c, brentq consideraba el
    intervalo "convergido" apenas se angostaba por debajo de ~2e-12 de
    ANCHO, mucho antes de haber localizado la raiz real con precision
    suficiente -- y devolvia el limite inferior del intervalo (c_lo) en vez
    de la raiz verdadera. Al buscar en log10(c), el ancho del intervalo de
    busqueda es solo ~13-17 unidades (no 1e-5), y la tolerancia por defecto
    de brentq ahi si es mas que suficiente para converger correctamente.

    Retorna None si no se encuentra una raiz en el intervalo [c_lo, c_hi].
    """
    def f_log(log10_c):
        c = 10.0 ** log10_c
        return delta_phi_de_c(c, Kd) - LOD_phi_objetivo

    log_lo, log_hi = np.log10(c_lo), np.log10(c_hi)

    # Verificar que haya cambio de signo en el intervalo (condicion de brentq)
    if f_log(log_lo) * f_log(log_hi) > 0:
        return None

    log_c_LOD = brentq(f_log, log_lo, log_hi, xtol=1e-10, rtol=1e-12)
    return 10.0 ** log_c_LOD

resultados_LOD = {}
for nombre, info in sgRNAs_info.items():
    c_LOD_i = encontrar_c_LOD(info["Kd_mut"], LOD_phi)
    SNR_1fM = delta_phi_de_c(1e-15, info["Kd_mut"]) / sigma_blank
    SNR_10nM = delta_phi_de_c(10e-9, info["Kd_mut"]) / sigma_blank
    resultados_LOD[nombre] = {"c_LOD": c_LOD_i, "SNR_1fM": SNR_1fM, "SNR_10nM": SNR_10nM}

print("\n" + "=" * 70)
print("RESULTADOS DE LOD (PRELIMINARES -- placeholder S_phase y sigma_blank)")
print("=" * 70)
print(f"{'sgRNA':<18s}{'LOD [M]':>14s}{'LOD [pM]':>12s}{'SNR@1fM':>10s}{'SNR@10nM':>10s}")
for nombre, r in resultados_LOD.items():
    if r["c_LOD"] is not None:
        lod_txt = f"{r['c_LOD']:.2e}"
        lod_pM_txt = f"{r['c_LOD']*1e12:.3f}"
    else:
        lod_txt = "fuera rango"
        lod_pM_txt = "--"
    print(f"{nombre:<18s}{lod_txt:>14s}{lod_pM_txt:>12s}{r['SNR_1fM']:>10.2f}{r['SNR_10nM']:>10.2f}")


# =======================================================================
# SECCION 8: FACTOR DE DISCRIMINACION EN SENAL DE FASE
# =======================================================================
c_test = 1e-9   # [M] 1 nM, concentracion de referencia
print("\nFACTOR DE DISCRIMINACION EN SENAL DE FASE (a [ctDNA]=1 nM):")
for nombre, info in sgRNAs_info.items():
    phi_mut_test = delta_phi_de_c(c_test, info["Kd_mut"])
    phi_wt_test = delta_phi_de_c(c_test, info["Kd_wt"])
    discrim_phi = phi_mut_test / phi_wt_test if phi_wt_test > 0 else float("inf")
    print(f"  {nombre:<18s} Discriminacion_phi = {discrim_phi:.1f}x")
print("\nNOTA: la discriminacion en SENAL DE FASE es igual a la discriminacion en Gamma")
print("(porque Delta_phi = K x Gamma y K es el MISMO para mutante y silvestre -- ambos son")
print("ctDNA, misma MW, mismo dn/dc). El beneficio real del biosensor esta en que la")
print("CONCENTRACION necesaria para producir una senal dada es distinta entre mutante y")
print("silvestre, no en que la senal en si sea de naturaleza distinta.")
print("=" * 70)


# =======================================================================
# VISUALIZACION - FIGURA 1: FUNCION DE TRANSDUCCION (1x2)
# =======================================================================
fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(14, 6))
fig1.suptitle("Modulo F - Funcion de transduccion Gamma -> Delta_phi", fontsize=14, fontweight="bold")

Gamma_barrido = np.linspace(0, Gamma_max_ef, 100)
Delta_phi_barrido_B = delta_phi_ruta_B(Gamma_barrido)

ax1a.plot(Gamma_barrido, Delta_phi_barrido_B, color="tab:blue", linewidth=2)
ax1a.plot([Gamma_max_ef], [Delta_phi_max], "o", color="tab:red", markersize=9, zorder=5)
ax1a.annotate(f"K_transduction = {K_transduction:.2e} deg*mm^2/mol",
              xy=(0.05, 0.90), xycoords="axes fraction", fontsize=9)
wrap_txt = "sin phase wrapping" if Delta_phi_max < 180 else "CON phase wrapping"
ax1a.annotate(f"Delta_phi_max = {Delta_phi_max:.3e} deg ({wrap_txt})",
              xy=(0.05, 0.83), xycoords="axes fraction", fontsize=9)
ax1a.set_xlabel("Gamma (mol/mm^2)")
ax1a.set_ylabel("Delta_phi (deg)")
ax1a.set_title("Funcion de transduccion: Gamma -> Delta_phi (Ruta B)")
ax1a.grid(alpha=0.3)

Gamma_seis = np.linspace(0, Gamma_max_ef, 6)
phi_A_seis = delta_phi_ruta_A(Gamma_seis)
phi_B_seis = delta_phi_ruta_B(Gamma_seis)
ax1b.plot(Gamma_seis, phi_A_seis, "o-", color="tab:orange", linewidth=2, markersize=8, label="Ruta A (via RU)")
ax1b.plot(Gamma_seis, phi_B_seis, "s--", color="tab:blue", linewidth=2, markersize=8, label="Ruta B (dn/dc explicito)")
ax1b.annotate(f"Diferencia Ruta A vs B: {diferencia_rutas_pct:.1f}%",
              xy=(0.05, 0.90), xycoords="axes fraction", fontsize=9,
              color="tab:red" if diferencia_rutas_pct > 20 else "black")
ax1b.set_xlabel("Gamma (mol/mm^2)")
ax1b.set_ylabel("Delta_phi (deg)")
ax1b.set_yscale("log")
ax1b.set_title("Cross-validacion: RU vs. dn/dc explicito")
ax1b.legend(fontsize=9)
ax1b.grid(alpha=0.3, which="both")

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("ModuloF_funcion_transduccion.png", dpi=150)
print("\nFigura guardada como: ModuloF_funcion_transduccion.png")


# =======================================================================
# VISUALIZACION - FIGURA 2: DOSIS-RESPUESTA EN SENAL DE FASE (2x3)
# =======================================================================
fig2, axes2 = plt.subplots(2, 3, figsize=(19, 10))
fig2.suptitle("Modulo F - Senal SPR de Fase vs [ctDNA] (3 sgRNAs)\n"
              "S_phase y sigma_blank son PLACEHOLDERS -- LOD no definitivo",
              fontsize=13, fontweight="bold")

nombres_sgRNA_F = list(sgRNAs_info.keys())

for col, nombre in enumerate(nombres_sgRNA_F):
    info = sgRNAs_info[nombre]
    c_LOD_i = resultados_LOD[nombre]["c_LOD"]

    for fila, (xmin_local, xmax_local, titulo_extra) in enumerate([(1e-15, 1e-7, "vista completa"), (1e-15, 1e-8, "ZOOM rango clinico")]):
        ax = axes2[fila, col]
        mascara = (c_array >= xmin_local) & (c_array <= xmax_local)
        ax.plot(c_array[mascara], info["phi_mut"][mascara], color="tab:blue", linewidth=2,
                label=f"Mutante (Kd={info['Kd_mut']*1e9:.2f} nM)")
        ax.plot(c_array[mascara], info["phi_wt"][mascara], color="tab:red", linewidth=2, linestyle=":",
                label=f"Silvestre (Kd={info['Kd_wt']*1e9:.1f} nM)")
        ax.axhline(LOD_phi, color="tab:orange", linestyle="--", linewidth=1.3,
                   label=f"LOD=3sigma_blank={LOD_phi:.1e} deg (PLACEHOLDER)")
        if c_LOD_i is not None and xmin_local <= c_LOD_i <= xmax_local:
            ax.axvline(c_LOD_i, color="gray", linestyle=":", linewidth=1.2)
        ax.axvspan(max(1e-15, xmin_local), min(1e-8, xmax_local), color="gray", alpha=0.08)
        ax.set_xscale("log")
        ax.set_xlabel("[ctDNA] (M)")
        ax.set_ylabel("Delta_phi (deg)")
        ax.set_title(f"{nombre} ({titulo_extra})")
        if col == 0:
            ax.legend(fontsize=6.5, loc="upper left")
        ax.grid(alpha=0.3, which="both")

plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig("ModuloF_dosis_respuesta_fase.png", dpi=150)
print("Figura guardada como: ModuloF_dosis_respuesta_fase.png")


# =======================================================================
# VISUALIZACION - FIGURA 3: LOD COMPARATIVO Y SNR (1x2)
# =======================================================================
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 6))
fig3.suptitle("Modulo F - LOD comparativo y relacion senal-ruido", fontsize=14, fontweight="bold")

nombres_barra_F = nombres_sgRNA_F
colores_barra_F = ["tab:blue", "tab:green", "tab:purple"]
log_LOD = []
for nombre in nombres_barra_F:
    c_LOD_i = resultados_LOD[nombre]["c_LOD"]
    log_LOD.append(np.log10(c_LOD_i) if c_LOD_i is not None else np.nan)

ax3a.barh(nombres_barra_F, log_LOD, color=colores_barra_F, edgecolor="black", alpha=0.85)
for ref_val, ref_label in [(-15, "1 fM"), (-12, "1 pM"), (-9, "1 nM")]:
    ax3a.axvline(ref_val, color="gray", linestyle=":", linewidth=1)
    ax3a.text(ref_val, -0.7, ref_label, fontsize=8, color="gray", ha="center")
ax3a.set_xlabel("log10([ctDNA]_LOD) (M)")
ax3a.set_title("LOD de deteccion para ctDNA mutante (3 sgRNAs)")
ax3a.text(0.02, 0.02, "LOD = concentracion donde Delta_phi_mut = 3 sigma_blank\n"
                        "LOD calculado con sigma_blank PLACEHOLDER",
          transform=ax3a.transAxes, fontsize=7.5, style="italic", va="bottom")
ax3a.grid(alpha=0.3, axis="x")

x_pos = np.arange(len(nombres_barra_F))
ancho = 0.35
snr_1fM_vals = [resultados_LOD[n]["SNR_1fM"] for n in nombres_barra_F]
snr_10nM_vals = [resultados_LOD[n]["SNR_10nM"] for n in nombres_barra_F]
log_snr_1fM = np.log10(np.maximum(snr_1fM_vals, 1e-10))
log_snr_10nM = np.log10(np.maximum(snr_10nM_vals, 1e-10))

ax3b.bar(x_pos - ancho/2, log_snr_1fM, ancho, label="SNR @ 1 fM", color="tab:cyan", edgecolor="black")
ax3b.bar(x_pos + ancho/2, log_snr_10nM, ancho, label="SNR @ 10 nM", color="tab:brown", edgecolor="black")
ax3b.axhline(np.log10(3), color="red", linestyle="--", linewidth=1.3, label="Umbral SNR=3")
ax3b.set_xticks(x_pos)
ax3b.set_xticklabels(nombres_barra_F, fontsize=8)
ax3b.set_ylabel("log10(SNR)")
ax3b.set_title("Relacion senal-ruido en extremos del rango clinico")
ax3b.legend(fontsize=8)
ax3b.grid(alpha=0.3, axis="y")
if any(v < 3 for v in snr_1fM_vals):
    ax3b.text(0.5, 0.02, "No detectable a 1 fM con parametros actuales (SNR<3 para al menos un sgRNA)",
              transform=ax3b.transAxes, ha="center", fontsize=7.5, color="red", style="italic")

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("ModuloF_LOD_comparativo.png", dpi=150)
print("Figura guardada como: ModuloF_LOD_comparativo.png")


# =======================================================================
# VISUALIZACION - FIGURA 4: SENSIBILIDAD A PARAMETROS WETLAB (2x2)
# =======================================================================
fig4 = plt.figure(figsize=(15, 11))
gs4 = gridspec.GridSpec(2, 2, figure=fig4)
fig4.suptitle("Modulo F - Sensibilidad del LOD a parametros pendientes de WetLab", fontsize=14, fontweight="bold")

ax4a = fig4.add_subplot(gs4[0, 0])
ax4b = fig4.add_subplot(gs4[0, 1])
ax4c = fig4.add_subplot(gs4[1, 0])
ax4d = fig4.add_subplot(gs4[1, 1])

Kd_ref = Kd_EGFR19_mut   # "mejor sgRNA" (mayor discriminacion, Kd mas bajo)

def K_B_de_Sphase(S_phase_val):
    return S_phase_val * dn_dc_DNA * MW_ctDNA / d_eff

def LOD_conc_aproximado(K_B_val, sigma_blank_val, Kd_val):
    """
    Aproximacion analitica (inversa de Langmuir) del LOD, usada para barridos
    rapidos en grillas grandes. LOD_phi = 3*sigma_blank_val;
    Delta_phi_max = K_B_val * Gamma_max_ef.
    """
    LOD_phi_val = 3.0 * sigma_blank_val
    Delta_phi_max_val = K_B_val * Gamma_max_ef
    if Delta_phi_max_val <= LOD_phi_val:
        return np.nan   # nunca se alcanza el LOD, ni saturando el chip
    return Kd_val * LOD_phi_val / (Delta_phi_max_val - LOD_phi_val)

# --- Panel (1,1): LOD vs S_phase ---
S_phase_barrido = np.logspace(2, 5, 5)
LOD_vs_Sphase = [LOD_conc_aproximado(K_B_de_Sphase(s), sigma_blank, Kd_ref) for s in S_phase_barrido]
ax4a.plot(S_phase_barrido, LOD_vs_Sphase, "o-", color="tab:blue", linewidth=2, markersize=8)
idx_placeholder = np.argmin(np.abs(S_phase_barrido - S_phase))
ax4a.plot(S_phase_barrido[idx_placeholder], LOD_vs_Sphase[idx_placeholder], "*", color="tab:red", markersize=18, zorder=5, label="Placeholder actual")
ax4a.set_xscale("log")
ax4a.set_yscale("log")
ax4a.set_xlabel("S_phase (deg/RIU)")
ax4a.set_ylabel("LOD (M)")
ax4a.set_title("LOD vs. S_phase (EGFR exon19del)")
ax4a.legend(fontsize=8)
ax4a.grid(alpha=0.3, which="both")

# --- Panel (1,2): LOD vs sigma_blank ---
sigma_barrido = np.logspace(-6, -2, 5)
LOD_vs_sigma = [LOD_conc_aproximado(K_transduction_B, s, Kd_ref) for s in sigma_barrido]
ax4b.plot(sigma_barrido, LOD_vs_sigma, "o-", color="tab:green", linewidth=2, markersize=8)
idx_placeholder_sigma = np.argmin(np.abs(sigma_barrido - sigma_blank))
ax4b.plot(sigma_barrido[idx_placeholder_sigma], LOD_vs_sigma[idx_placeholder_sigma], "*", color="tab:red", markersize=18, zorder=5, label="Placeholder actual")
ax4b.set_xscale("log")
ax4b.set_yscale("log")
ax4b.set_xlabel("sigma_blank (deg)")
ax4b.set_ylabel("LOD (M)")
ax4b.set_title("LOD vs. sigma_blank (EGFR exon19del)")
ax4b.legend(fontsize=8)
ax4b.grid(alpha=0.3, which="both")

# --- Panel (2,1): Delta_phi_max vs S_phase (verificacion phase wrapping) ---
S_phase_fino = np.logspace(2, 6, 200)
Delta_phi_max_vs_S = K_B_de_Sphase(S_phase_fino) * Gamma_max_ef
ax4c.plot(S_phase_fino, Delta_phi_max_vs_S, color="black", linewidth=2)
ax4c.axhline(180, color="red", linestyle="--", linewidth=1.3)
ax4c.fill_between(S_phase_fino, 180, np.maximum(Delta_phi_max_vs_S.max(), 180) * 1.1,
                   color="red", alpha=0.12)
ax4c.fill_between(S_phase_fino, 0, 180, color="green", alpha=0.10)
ax4c.text(0.05, 0.92, "Regimen lineal valido", transform=ax4c.transAxes, fontsize=8, color="darkgreen")
ax4c.text(0.05, 0.5, "Phase wrapping", transform=ax4c.transAxes, fontsize=8, color="darkred")
ax4c.axvline(S_phase, color="tab:red", linestyle=":", linewidth=1.3, label="Placeholder actual")
ax4c.set_xscale("log")
ax4c.set_xlabel("S_phase (deg/RIU)")
ax4c.set_ylabel("Delta_phi_max (deg)")
ax4c.set_title("Verificacion de no-wrapping vs. S_phase")
ax4c.legend(fontsize=8, loc="upper left")
ax4c.grid(alpha=0.3, which="both")

# --- Panel (2,2): Mapa de LOD 2D (S_phase vs sigma_blank) ---
n_grid = 20
S_grid = np.logspace(2, 5, n_grid)
sigma_grid = np.logspace(-6, -2, n_grid)
LOD_grid = np.full((n_grid, n_grid), np.nan)
for i, s_val in enumerate(sigma_grid):
    for j, S_val in enumerate(S_grid):
        lod_ij = LOD_conc_aproximado(K_B_de_Sphase(S_val), s_val, Kd_ref)
        LOD_grid[i, j] = np.log10(lod_ij) if (lod_ij is not None and not np.isnan(lod_ij) and lod_ij > 0) else np.nan

X_grid, Y_grid = np.meshgrid(np.log10(S_grid), np.log10(sigma_grid))
cf = ax4d.contourf(X_grid, Y_grid, LOD_grid, levels=20, cmap="viridis")
cbar = plt.colorbar(cf, ax=ax4d)
cbar.set_label("log10(LOD) (M)")
contornos_lineas = ax4d.contour(X_grid, Y_grid, LOD_grid, levels=[-15, -12, -10, -9], colors="white", linewidths=1)
ax4d.clabel(contornos_lineas, fmt=lambda v: f"{10**v:.0e} M", fontsize=7)
ax4d.plot(np.log10(S_phase), np.log10(sigma_blank), "o", color="cyan", markersize=12,
          markeredgecolor="black", zorder=5)
ax4d.set_xlabel("log10(S_phase) (deg/RIU)")
ax4d.set_ylabel("log10(sigma_blank) (deg)")
ax4d.set_title("Mapa de LOD -- EGFR exon19del sgRNA")
ax4d.text(0.02, -0.18, "El punto azul es el PLACEHOLDER actual.\nDonde caiga su sistema real determinara el LOD.",
          transform=ax4d.transAxes, fontsize=7.5, style="italic")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("ModuloF_sensibilidad_WetLab.png", dpi=150)
print("Figura guardada como: ModuloF_sensibilidad_WetLab.png")

plt.show()


# =======================================================================
# OUTPUT EN CONSOLA FINAL
# =======================================================================
print("\n" + "=" * 70)
print("SENAL A CONCENTRACIONES CLINICAS CLAVE")
print("(sgRNA = EGFR exon19del, Kd_mut = 0.5 nM)")
print("=" * 70)
for c_ref, etiqueta in [(1e-15, "1 fM"), (1e-12, "1 pM"), (1e-9, "1 nM")]:
    phi_ref = delta_phi_de_c(c_ref, Kd_EGFR19_mut)
    snr_ref = phi_ref / sigma_blank
    print(f"  [ctDNA] = {etiqueta}:  Delta_phi = {phi_ref:.3e} deg, SNR = {snr_ref:.2f}")

print("\n" + "=" * 70)
print("DISCLAIMER -- LEER ANTES DE USAR ESTOS RESULTADOS")
print("=" * 70)
print("Los valores de LOD calculados en este modulo son PRELIMINARES.")
print("Dependen criticamente de S_phase y sigma_blank, que son PLACEHOLDERS")
print("tomados de literatura para sistemas SPR distintos al de su laboratorio.")
print("El LOD real puede diferir en ordenes de magnitud segun el instrumento")
print("especifico.")
print("\nPara obtener LOD definitivo:")
print("  1. Medir S_phase: correr soluciones de indice conocido (glicerol/sacarosa)")
print("     y registrar Delta_phi vs Delta_n (calculado de tablas). Pendiente = S_phase.")
print("  2. Medir sigma_blank: >=20 corridas de buffer solo, calcular std(Delta_phi).")
print("  3. Actualizar ambos parametros en este script.")
print("  4. Re-correr para obtener LOD real del sistema.")
print("=" * 70)

mejor_sgRNA = min(
    (n for n in resultados_LOD if resultados_LOD[n]["c_LOD"] is not None),
    key=lambda n: resultados_LOD[n]["c_LOD"],
    default=None,
)

print("OUTPUT PARA MODULO G:")
print("  Funcion: Delta_phi(c) = K_transduction x Gamma_max_ef x c/(Kd+c)")
print(f"  K_transduction          = {K_transduction:.3e} deg*mm^2/mol   (Ruta B)")
print(f"  sigma_blank             = {sigma_blank:.1e} deg            (PLACEHOLDER)")
print(f"  LOD_phi (3sigma)        = {LOD_phi:.1e} deg            (PLACEHOLDER)")
print(f"  Mejor sgRNA             = {mejor_sgRNA} (LOD_conc mas bajo)")
print("=" * 70)

# Diccionario final con todos los resultados, para facilitar importacion por Modulo G
resultados_modulo_F = {
    "K_transduction": K_transduction,
    "K_transduction_A": K_transduction_A,
    "K_transduction_B": K_transduction_B,
    "diferencia_rutas_pct": diferencia_rutas_pct,
    "Delta_phi_max": Delta_phi_max,
    "S_phase": S_phase,
    "sigma_blank": sigma_blank,
    "LOD_phi": LOD_phi,
    "resultados_LOD_por_sgRNA": resultados_LOD,
    "mejor_sgRNA": mejor_sgRNA,
    "Gamma_max_ef": Gamma_max_ef,
    "sgRNAs_info": sgRNAs_info,
}


# =======================================================================
# SECCION DE INTEGRACION CON MODULOS
# =======================================================================
#
#   1. Cuando WetLab mida S_phase y sigma_blank REALES (calibracion con
#      soluciones de indice conocido + >=20 corridas de blanco), esos dos
#      valores reemplazan los PLACEHOLDERS de la Seccion D de este script.
#      Todo lo demas (K_transduction, curvas dosis-respuesta, LOD) se
#      recalcula automaticamente.
#
#   2. K_transduction y la funcion Delta_phi(c) = K_transduction x
#      Gamma_max_ef x c/(Kd+c) pasan al Modulo G, que combinara esta
#      funcion con sigma_blank para calcular el LOD final formal del
#      sistema completo (con el analisis estadistico que corresponda,
#      mas alla del criterio simple 3-sigma usado aqui como aproximacion).
#
#   3. Si Delta_phi_max > 90 grados (aunque no haya wrapping formal en
#      180 grados), la aproximacion lineal Delta_phi = K x Gamma puede
#      empezar a perder precision -- en ese caso, considerar el modelo
#      completo de reflectividad de Fresnel para la curva fase-angulo,
#      en vez de la aproximacion lineal usada en este modulo.
#
#   4. La DISCRIMINACION mutante/wildtype en CONCENTRACION (es decir, que
#      LOD_conc del mutante sea mucho menor que la concentracion a la que
#      el silvestre empieza a dar senal significativa) es el output mas
#      importante de todo este modulo para evaluar la especificidad
#      clinica del biosensor -- NO la discriminacion en la magnitud de la
#      senal en si, que es identica para mutante y silvestre (mismo K).
#
# =======================================================================