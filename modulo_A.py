#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=======================================================================
MODULO A v2.0 - MODELO UNIFICADO DE EQUIPO (Damian + Valeria)
=======================================================================
Produccion de dCas9-AviTag y BirA en E. coli BL21(DE3) - Sistema de 5 ODEs
Proyecto iGEM 2025 - Deteccion de mutaciones oncogenicas en ctDNA

Este script REEMPLAZA a modulo_A_expresion_genica.py (v1.0). Es el
resultado de fusionar los modelos independientes de Damian y Valeria
tras una auditoria bibliografica formal (ver
Comparacion_ModuloA_Damian_vs_Valeria.pdf). Tres citas de v1.0 fueron
eliminadas por incorrectas o no verificables (ver Seccion de parametros
mas abajo, buscar "CAMBIOS RESPECTO A v1.0" en cada parametro afectado).

# ============================================================
# INSTALACION DE DEPENDENCIAS (ejecutar en terminal una sola vez)
# ============================================================
# conda activate iGEM
# conda install numpy scipy matplotlib
#
# ============================================================
"""

# =======================================================================
# IMPORTS
# =======================================================================
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

np.random.seed(42)  # reproducibilidad de las corridas Monte Carlo


# =======================================================================
# SECCION 1 - CLASE Param
# =======================================================================
class Param:
    """
    Representa un parametro numerico del modelo con metadatos de
    trazabilidad QbD (Quality by Design).

    origin permitido: 'LIT' | 'WETLAB_PEND' | 'PLACEHOLDER' | 'DERIVED' | 'PROTOCOL'
    """

    def __init__(self, value, unit, origin, citation, rel_unc,
                 sampling="triangular", wet_lab_note=None):
        self.value = value
        self.unit = unit
        self.origin = origin
        self.citation = citation
        self.rel_unc = rel_unc
        self.sampling = sampling
        self.wet_lab_note = wet_lab_note

    def sample(self):
        """Muestrea un valor para Monte Carlo respetando el metodo de muestreo."""
        if self.rel_unc <= 0:
            return self.value
        lo = self.value * (1 - self.rel_unc)
        hi = self.value * (1 + self.rel_unc)
        if self.sampling == "triangular":
            return np.random.triangular(lo, self.value, hi)
        elif self.sampling == "uniform":
            return np.random.uniform(lo, hi)
        else:
            return self.value

    def __repr__(self):
        return f"Param({self.value} {self.unit}, origin={self.origin}, rel_unc={self.rel_unc:.0%})"


# =======================================================================
# SECCION 2 - FUNCION Q10
# =======================================================================
def ajustar_Q10(k_ref, Q10, T_exp=16.0, T_ref=37.0):
    """Corrige una constante cinetica de T_ref a T_exp usando el factor Q10."""
    return k_ref * (Q10 ** ((T_exp - T_ref) / 10.0))


# =======================================================================
# SECCION 3 - PARAMETROS (todos como instancias de Param)
# =======================================================================

# --- TEMPERATURAS ---
T_exp = Param(16.0, "C", "PROTOCOL",
              "Temperatura de induccion del protocolo iGEM 2025. Estrategia cold "
              "shock para mejorar plegamiento de dCas9.", rel_unc=0.0)

T_ref = Param(37.0, "C", "PROTOCOL",
              "Temperatura de referencia estandar para parametros de E. coli en literatura.",
              rel_unc=0.0)

# --- INDUCCION IPTG (funcion de Hill, adoptado del modelo de Valeria) ---
IPTG_conc = Param(0.5e-3, "M", "PROTOCOL",
                   "Concentracion de IPTG usada en el protocolo de induccion del equipo. "
                   "Documentada como saturante para BL21(DE3) a esta concentracion.",
                   rel_unc=0.0)

K_IPTG = Param(0.1e-3, "M", "PLACEHOLDER",
               "Constante de disociacion de LacI-IPTG. PLACEHOLDER: valor representativo "
               "de la literatura, sin cita primaria verificada en ninguno de los dos "
               "modelos del equipo. Referencia tentativa: Lewis, M. (2005). The lac "
               "repressor. Comptes Rendus Biologies, 328(6), 521-548. "
               "https://doi.org/10.1016/j.crvi.2005.02.004",
               rel_unc=0.30,
               wet_lab_note="Confirmar con curva de dosis-respuesta IPTG vs. expresion de reportero.")

n_Hill = Param(2.0, "adim", "LIT",
               "Coeficiente de Hill para cooperatividad de LacI (dimero de dimeros). "
               "Fuente: Oehler, S., et al. (1994). The three operators of the lac operon "
               "cooperate in repression. EMBO Journal, 13(14), 3348-3355. "
               "https://doi.org/10.1002/j.1460-2075.1994.tb06637.x",
               rel_unc=0.10)

# --- SINTESIS DE T7 RNAP ---
# CAMBIO vs v1.0: se elimina la cita de Damian (Carrier & Keasling 1999, que trata
# sobre estabilidad de ARNm, no sintesis de T7 RNAP). Se marca WETLAB_PEND con
# rango plausible [5-50] nM/h, valor central 20.0.
k_R_ref = Param(20.0, "nM/h", "WETLAB_PEND",
                 "WETLAB_PEND: ninguno de los dos modelos del equipo tiene cita primaria "
                 "verificable para este parametro. El modelo de Damian citaba incorrectamente "
                 "a Carrier & Keasling (1999), que trata sobre estabilidad de ARNm (no sintesis "
                 "de T7 RNAP). El modelo de Valeria usaba 50.0 sin cita. Rango plausible: "
                 "5-50 nM/h, con valor central de 20 nM/h. Para calibracion, ver: Studier, "
                 "F. W., & Moffatt, B. A. (1986). Use of bacteriophage T7 RNA polymerase to "
                 "direct selective high-level expression. Journal of Molecular Biology, "
                 "189(1), 113-130. https://doi.org/10.1016/0022-2836(86)90385-2",
                 rel_unc=0.60, sampling="uniform",
                 wet_lab_note="Medir con Western blot cuantitativo de T7 RNAP vs. tiempo "
                              "tras induccion con IPTG, o con reportero fluorescente bajo "
                              "promotor lacUV5 calibrado a concentracion absoluta (nM). "
                              "PARAMETRO CON MAYOR IMPACTO EN LA DISCREPANCIA ENTRE MODELOS.")

Q10_tx = Param(1.8, "adim", "LIT",
               "Q10 para transcripcion por T7 RNAP. Fuente: Chamberlin, M., & Ring, J. "
               "(1973). Characterization of T7-specific ribonucleic acid polymerase. "
               "Journal of Biological Chemistry, 248(6), 2235-2244. "
               "https://doi.org/10.1016/S0021-9258(19)44178-4 "
               "NOTA: este paper caracteriza T7 RNAP in vitro; el Q10 in vivo podria diferir.",
               rel_unc=0.20)

delta_R_ref = Param(0.2, "h^-1", "LIT",
                     "Tasa de degradacion de T7 RNAP a 37C. Vida media ~3.5h. NOTA: la cita "
                     "original (Perez-Perez & Gutierrez 1995) trata sobre un vector de "
                     "expresion con arabinosa y no mide directamente la degradacion de T7 "
                     "RNAP. Ambos modelos del equipo coincidieron en 0.20 h^-1 de forma "
                     "independiente, lo que da cierta robustez al valor aunque la cita sea "
                     "debil. Fuente de referencia general: Goldberg, A. L. (2003). Protein "
                     "degradation and protection against misfolded or damaged proteins. "
                     "Nature, 426(6968), 895-899. https://doi.org/10.1038/nature02263",
                     rel_unc=0.30)

Q10_delta_R = Param(1.5, "adim", "LIT",
                     "Q10 para degradacion de proteinas estables (proteasas). Fuente: "
                     "Goldberg, A. L. (2003). Nature, 426(6968), 895-899. "
                     "https://doi.org/10.1038/nature02263",
                     rel_unc=0.20)

# --- TRANSCRIPCION (T_L y T_2) ---
k_tx_ref = Param(2.0, "h^-1 nM^-1", "LIT",
                  "Tasa de transcripcion por molecula de T7 RNAP en E. coli in vivo. "
                  "Fuente: Golding, I., Paulsson, J., Zawilski, S. M., & Cox, E. C. (2005). "
                  "Real-time kinetics of gene activity in individual bacteria. Cell, "
                  "123(6), 1025-1036. https://doi.org/10.1016/j.cell.2005.09.031 "
                  "NOTA: el modelo de Valeria usaba 8.0 sin cita (4x mayor). Diferencia no "
                  "resuelta; este valor (2.0) tiene respaldo bibliografico verificable.",
                  rel_unc=0.40)

delta_m_ref = Param(6.0, "h^-1", "LIT",
                     "Tasa de degradacion de ARNm en E. coli a 37C (vida media ~5-8 min). "
                     "AMBOS modelos del equipo coincidieron en este valor de forma "
                     "independiente. Fuente: Bernstein, J. A., Khodursky, A. B., Lin, P. H., "
                     "Lin-Chao, S., & Cohen, S. N. (2002). Global analysis of mRNA decay and "
                     "abundance in Escherichia coli at single-gene resolution using two-color "
                     "fluorescent DNA microarrays. PNAS, 99(15), 9697-9702. "
                     "https://doi.org/10.1073/pnas.112318199",
                     rel_unc=0.20)

Q10_delta_m = Param(2.0, "adim", "LIT",
                     "Q10 para degradacion de ARNm (RNasas). Proceso con mayor sensibilidad "
                     "termica que proteasas -- justifica Q10 diferenciado. Fuente: Phadtare, "
                     "S., & Severinov, K. (2010). RNA remodeling and gene regulation by cold "
                     "shock proteins. RNA Biology, 7(6), 788-795. "
                     "https://doi.org/10.4161/rna.7.6.13482",
                     rel_unc=0.20)

# --- TRADUCCION -- parametrizacion por RBS (adoptado del enfoque de Valeria) ---
k_tl0_ref = Param(3.5, "h^-1 nM^-1", "LIT",
                   "Tasa base de traduccion en E. coli (ribosoma estandar). Fuente: Salis, "
                   "H. M., Mirsky, E. A., & Voigt, C. A. (2009). Automated design of "
                   "synthetic ribosome binding sites to control protein expression. Nature "
                   "Biotechnology, 27(10), 946-950. https://doi.org/10.1038/nbt.1568",
                   rel_unc=0.30)

RBS_str_1 = Param(1.0, "adim", "WETLAB_PEND",
                   "Fuerza relativa del RBS1 (dCas9-AviTag) normalizada respecto a k_tl0. "
                   "Valor nominal 1.0 = fuerza estandar. WETLAB_PEND: calcular con RBS "
                   "Calculator v2.1 (Salis Lab, https://salislab.net/software/) usando la "
                   "secuencia real del RBS1 de la construccion. El TIR calculado se "
                   "convierte a RBS_str dividiendo por el TIR de la secuencia de referencia.",
                   rel_unc=0.30,
                   wet_lab_note="Ingresar secuencia real del RBS1 en RBS Calculator para obtener TIR.")

RBS_str_2 = Param(0.57, "adim", "DERIVED",
                   "Fuerza relativa del RBS2 (BirA) = k_tl2_v1 / k_tl1_v1 = 2.0 / 3.5. "
                   "Derivado de los valores usados en el modelo v1.0 de Damian. "
                   "WETLAB_PEND: calibrar con RBS Calculator igual que RBS_str_1.",
                   rel_unc=0.30,
                   wet_lab_note="Ingresar secuencia real del RBS2 en RBS Calculator para obtener TIR.")

Q10_tl = Param(2.2, "adim", "LIT",
               "Q10 para traduccion (actividad ribosomal) en E. coli. Fuente: Broeze, R. J., "
               "Solomon, C. J., & Pope, D. H. (1978). Effects of low temperature on in vivo "
               "and in vitro protein synthesis in Escherichia coli and Pseudomonas "
               "fluorescens. Journal of Bacteriology, 134(3), 861-874. "
               "https://doi.org/10.1128/jb.134.3.861-874.1978",
               rel_unc=0.15)

# --- DEGRADACION DE PROTEINAS (diferenciada, adoptado de Valeria) ---
delta_p_dCas9_ref = Param(0.05, "h^-1", "LIT",
                           "Tasa de degradacion de dCas9-AviTag a 37C (vida media ~14h). "
                           "Proteina estable: la degradacion es lenta. AMBOS modelos del "
                           "equipo coincidieron en este valor de forma independiente. "
                           "Fuente: Maurizi, M. R. (1992). Proteases and protein degradation "
                           "in Escherichia coli. Experientia, 48(2), 178-201. "
                           "https://doi.org/10.1007/BF01923511",
                           rel_unc=0.30)

delta_p_BirA_ref = Param(0.08, "h^-1", "PLACEHOLDER",
                          "Tasa de degradacion de BirA a 37C. Vida media ~8.7h. PLACEHOLDER: "
                          "diferenciada de dCas9 siguiendo el modelo de Valeria (mas "
                          "granular), pero sin cita primaria verificada en ningun modelo. "
                          "Referencia general: Maurizi, M. R. (1992). Experientia, 48(2), "
                          "178-201. https://doi.org/10.1007/BF01923511",
                          rel_unc=0.30,
                          wet_lab_note="Medir degradacion de BirA por pulse-chase o Western "
                                       "blot en ausencia de sintesis nueva (anadiendo rifampicina).")

Q10_delta_p = Param(1.5, "adim", "LIT",
                     "Q10 para degradacion de proteinas (proteasas). Proceso con menor "
                     "sensibilidad termica que RNasas -- justifica Q10 diferenciado respecto "
                     "a degradacion de ARNm. Fuente: Goldberg, A. L. (2003). Nature, "
                     "426(6968), 895-899. https://doi.org/10.1038/nature02263",
                     rel_unc=0.20)

# --- CRECIMIENTO CELULAR ---
mu_ref = Param(1.386, "h^-1", "LIT",
               "Tasa de crecimiento de E. coli a 37C en LB (t_duplicacion ~30 min). Fuente: "
               "Farewell, A., & Neidhardt, F. C. (1998). Effect of temperature on in vivo "
               "protein synthetic capacity in Escherichia coli. Journal of Bacteriology, "
               "180(17), 4704-4710. https://doi.org/10.1128/JB.180.17.4704-4710.1998",
               rel_unc=0.10,
               wet_lab_note="Medir curva de crecimiento OD600 a 16C post-induccion para "
                            "obtener mu(16C) real.")

Q10_mu = Param(2.5, "adim", "LIT",
               "Q10 para tasa de crecimiento de E. coli. Fuente: Farewell, A., & Neidhardt, "
               "F. C. (1998). Journal of Bacteriology, 180(17), 4704-4710. "
               "https://doi.org/10.1128/JB.180.17.4704-4710.1998",
               rel_unc=0.20)

# --- PARAMETROS DE DISENO GENETICO (los mas inciertos del sistema) ---
# CAMBIO vs v1.0: se elimina la cita fabricada "Masulis et al. 2015" de Damian.
f_rt = Param(0.20, "adim", "WETLAB_PEND",
             "Eficiencia de read-through del Promotor T7 #1 hacia la ORF de BirA. "
             "WETLAB_PEND: ninguno de los dos modelos tiene cita primaria verificada. "
             "Damian citaba \"Masulis et al. (2015)\" -- paper NO encontrado en ninguna "
             "base de datos (posible cita fabricada). Valeria marco honestamente como "
             "WETLAB_PEND con valor 0.40. Rango plausible en sistemas T7 sin terminador "
             "intermedio: 0.10-0.50. Valor central de 0.20 usado como punto de partida. "
             "Para referencia general sobre read-through T7: Peters, J. M., et al. (2012). "
             "Rho and NusG suppress pervasive antisense transcription in Escherichia coli. "
             "Genes & Development, 26(23), 2621-2633. https://doi.org/10.1101/gad.196741.112",
             rel_unc=0.50, sampling="uniform",
             wet_lab_note="Medir por RT-qPCR comparando abundancia relativa de la region 3' "
                          "de dCas9 vs. region de BirA en el ARNm total del cultivo. Este es "
                          "el parametro de diseno mas incierto del modulo. "
                          "PRIORIDAD DE MEDICION: CRITICA.")

# CAMBIO vs v1.0: se mantiene phi_16 (unica cita primaria verificada del bloque).
phi_16 = Param(0.60, "adim", "LIT",
               "Fraccion de dCas9-AviTag correctamente plegada a 16C. AMBOS modelos "
               "convergieron de forma independiente (Damian: 0.60, Valeria: 0.65 -- "
               "diferencia del 8%). Fuente: Vera, A., Gonzalez-Montalban, N., Aris, A., & "
               "Villaverde, A. (2007). The conformational quality of insoluble recombinant "
               "proteins is enhanced at low growth temperatures. Biotechnology and "
               "Bioengineering, 96(6), 1101-1106. https://doi.org/10.1002/bit.21218 "
               "Este es el unico parametro de diseno con cita primaria verificable en "
               "ambos modelos.",
               rel_unc=0.15,
               wet_lab_note="Confirmar por SDS-PAGE: comparar fraccion soluble vs. "
                            "insoluble tras lisis del cultivo a 16C. PRIORIDAD: ALTA.")

# CAMBIO vs v1.0: se elimina la cita tematicamente incorrecta de Damian (Kusano et al. 1993).
phi_37 = Param(0.20, "adim", "WETLAB_PEND",
               "Fraccion de dCas9-AviTag correctamente plegada a 37C (referencia). "
               "WETLAB_PEND: Damian citaba incorrectamente a Kusano et al. (1993), paper "
               "que trata sobre factores sigma de la ARN polimerasa (sin relacion con "
               "plegamiento de proteinas recombinantes). Valeria usaba 0.30 sin cita. "
               "Ninguno verificado. Valor 0.20 como punto medio del rango plausible "
               "[0.10-0.35]. Para referencia general: Baneyx, F., & Mujacic, M. (2004). "
               "Recombinant protein folding and misfolding in Escherichia coli. Nature "
               "Biotechnology, 22(11), 1399-1408. https://doi.org/10.1038/nbt1029",
               rel_unc=0.40, sampling="uniform",
               wet_lab_note="Medir por SDS-PAGE comparando fraccion soluble vs. insoluble "
                            "tras lisis a 37C. Indispensable para comparar con phi_16 y "
                            "cuantificar la mejora del cold shock.")

# Diccionario con todos los Param, para iterar en las tablas de output
TODOS_LOS_PARAMS = {
    "T_exp": T_exp, "T_ref": T_ref, "IPTG_conc": IPTG_conc, "K_IPTG": K_IPTG,
    "n_Hill": n_Hill, "k_R_ref": k_R_ref, "Q10_tx": Q10_tx, "delta_R_ref": delta_R_ref,
    "Q10_delta_R": Q10_delta_R, "k_tx_ref": k_tx_ref, "delta_m_ref": delta_m_ref,
    "Q10_delta_m": Q10_delta_m, "k_tl0_ref": k_tl0_ref, "RBS_str_1": RBS_str_1,
    "RBS_str_2": RBS_str_2, "Q10_tl": Q10_tl, "delta_p_dCas9_ref": delta_p_dCas9_ref,
    "delta_p_BirA_ref": delta_p_BirA_ref, "Q10_delta_p": Q10_delta_p, "mu_ref": mu_ref,
    "Q10_mu": Q10_mu, "f_rt": f_rt, "phi_16": phi_16, "phi_37": phi_37,
}


# =======================================================================
# SECCION 4 - SISTEMA DE ODEs
# =======================================================================
def construir_params_16C(iptg_conc, k_iptg, n_hill, k_r_ref, q10_tx, delta_r_ref,
                          q10_delta_r, k_tx_ref_v, delta_m_ref_v, q10_delta_m,
                          k_tl0_ref_v, rbs1, rbs2, q10_tl, delta_p_dcas9_ref_v,
                          delta_p_bira_ref_v, q10_delta_p, mu_ref_v, q10_mu,
                          f_rt_val, T_exp_v=16.0, T_ref_v=37.0):
    """
    Construye el diccionario de VALORES NUMERICOS (ya ajustados a T_exp_v)
    que usa la funcion de ODEs. Recibe valores nominales o muestreados
    (floats), nunca objetos Param -- eso permite reusar esta funcion tanto
    para la corrida determinista como para cada iteracion de Monte Carlo.
    """
    induction_factor = (iptg_conc ** n_hill) / (k_iptg ** n_hill + iptg_conc ** n_hill)

    k_R = ajustar_Q10(k_r_ref, q10_tx, T_exp_v, T_ref_v)
    delta_R = ajustar_Q10(delta_r_ref, q10_delta_r, T_exp_v, T_ref_v)
    k_tx = ajustar_Q10(k_tx_ref_v, q10_tx, T_exp_v, T_ref_v)
    delta_m = ajustar_Q10(delta_m_ref_v, q10_delta_m, T_exp_v, T_ref_v)

    k_tl1_ref_v = k_tl0_ref_v * rbs1
    k_tl2_ref_v = k_tl0_ref_v * rbs2
    k_tl1 = ajustar_Q10(k_tl1_ref_v, q10_tl, T_exp_v, T_ref_v)
    k_tl2 = ajustar_Q10(k_tl2_ref_v, q10_tl, T_exp_v, T_ref_v)

    delta_p_dCas9 = ajustar_Q10(delta_p_dcas9_ref_v, q10_delta_p, T_exp_v, T_ref_v)
    delta_p_BirA = ajustar_Q10(delta_p_bira_ref_v, q10_delta_p, T_exp_v, T_ref_v)

    mu = ajustar_Q10(mu_ref_v, q10_mu, T_exp_v, T_ref_v)

    return {
        "k_R": k_R, "induction_factor": induction_factor, "delta_R": delta_R,
        "k_tx": k_tx, "delta_m": delta_m,
        "k_tl1": k_tl1, "k_tl2": k_tl2,
        "delta_p_dCas9": delta_p_dCas9, "delta_p_BirA": delta_p_BirA,
        "mu": mu, "f_rt": f_rt_val,
    }


def sistema_odes(t, y, p):
    """
    Sistema de 5 ODEs. y = [R, T_L, T_2, P_dCas9, P_BirA].
    p es un diccionario de VALORES NUMERICOS ya ajustados a 16C
    (construido con construir_params_16C).
    """
    R = max(y[0], 0.0)
    T_L = max(y[1], 0.0)
    T_2 = max(y[2], 0.0)
    P_dCas9 = max(y[3], 0.0)
    P_BirA = max(y[4], 0.0)

    dR = p["k_R"] * p["induction_factor"] - (p["delta_R"] + p["mu"]) * R
    dT_L = p["k_tx"] * R - (p["delta_m"] + p["mu"]) * T_L
    dT_2 = p["k_tx"] * R - (p["delta_m"] + p["mu"]) * T_2
    dP_dCas9 = p["k_tl1"] * T_L - (p["delta_p_dCas9"] + p["mu"]) * P_dCas9
    dP_BirA = p["k_tl2"] * (p["f_rt"] * T_L + T_2) - (p["delta_p_BirA"] + p["mu"]) * P_BirA

    return [dR, dT_L, dT_2, dP_dCas9, dP_BirA]


def params_nominales_dict():
    """Diccionario de valores NOMINALES (no muestreados) para la corrida determinista."""
    return construir_params_16C(
        IPTG_conc.value, K_IPTG.value, n_Hill.value, k_R_ref.value, Q10_tx.value,
        delta_R_ref.value, Q10_delta_R.value, k_tx_ref.value, delta_m_ref.value,
        Q10_delta_m.value, k_tl0_ref.value, RBS_str_1.value, RBS_str_2.value,
        Q10_tl.value, delta_p_dCas9_ref.value, delta_p_BirA_ref.value, Q10_delta_p.value,
        mu_ref.value, Q10_mu.value, f_rt.value, T_exp.value, T_ref.value,
    )


def params_muestreados_dict():
    """Diccionario de valores MUESTREADOS (Monte Carlo) para una iteracion."""
    return construir_params_16C(
        IPTG_conc.sample() if IPTG_conc.rel_unc > 0 else IPTG_conc.value,
        K_IPTG.sample() if K_IPTG.rel_unc > 0 else K_IPTG.value,
        n_Hill.sample() if n_Hill.rel_unc > 0 else n_Hill.value,
        k_R_ref.sample() if k_R_ref.rel_unc > 0 else k_R_ref.value,
        Q10_tx.sample() if Q10_tx.rel_unc > 0 else Q10_tx.value,
        delta_R_ref.sample() if delta_R_ref.rel_unc > 0 else delta_R_ref.value,
        Q10_delta_R.sample() if Q10_delta_R.rel_unc > 0 else Q10_delta_R.value,
        k_tx_ref.sample() if k_tx_ref.rel_unc > 0 else k_tx_ref.value,
        delta_m_ref.sample() if delta_m_ref.rel_unc > 0 else delta_m_ref.value,
        Q10_delta_m.sample() if Q10_delta_m.rel_unc > 0 else Q10_delta_m.value,
        k_tl0_ref.sample() if k_tl0_ref.rel_unc > 0 else k_tl0_ref.value,
        RBS_str_1.sample() if RBS_str_1.rel_unc > 0 else RBS_str_1.value,
        RBS_str_2.sample() if RBS_str_2.rel_unc > 0 else RBS_str_2.value,
        Q10_tl.sample() if Q10_tl.rel_unc > 0 else Q10_tl.value,
        delta_p_dCas9_ref.sample() if delta_p_dCas9_ref.rel_unc > 0 else delta_p_dCas9_ref.value,
        delta_p_BirA_ref.sample() if delta_p_BirA_ref.rel_unc > 0 else delta_p_BirA_ref.value,
        Q10_delta_p.sample() if Q10_delta_p.rel_unc > 0 else Q10_delta_p.value,
        mu_ref.sample() if mu_ref.rel_unc > 0 else mu_ref.value,
        Q10_mu.sample() if Q10_mu.rel_unc > 0 else Q10_mu.value,
        f_rt.sample() if f_rt.rel_unc > 0 else f_rt.value,
        T_exp.value, T_ref.value,
    )


# =======================================================================
# SECCION 5 - SIMULACION DETERMINISTICA (corrida nominal)
# =======================================================================
Y0 = [0.0, 0.0, 0.0, 0.0, 0.0]
T_END = 20.0
t_eval_fino = np.linspace(0, T_END, 5000)

params_nom = params_nominales_dict()
sol_nominal = solve_ivp(
    fun=sistema_odes, t_span=(0, T_END), y0=Y0, method="Radau",
    t_eval=t_eval_fino, args=(params_nom,), rtol=1e-8, atol=1e-10,
)

print("=" * 90)
print("MODULO A v2.0 - Modelo Unificado de Equipo (Damian + Valeria)")
print("=" * 90)
if sol_nominal.success:
    print("Integracion determinista: EXITOSA (Radau convergio correctamente)")
else:
    print(f"ADVERTENCIA: integracion determinista NO convergio: {sol_nominal.message}")

R_t = np.maximum(sol_nominal.y[0], 0.0)
T_L_t = np.maximum(sol_nominal.y[1], 0.0)
T_2_t = np.maximum(sol_nominal.y[2], 0.0)
P_dCas9_t = np.maximum(sol_nominal.y[3], 0.0)
P_BirA_t = np.maximum(sol_nominal.y[4], 0.0)
t_horas = sol_nominal.t

P_dCas9_20h_det = P_dCas9_t[-1]
P_BirA_20h_det = P_BirA_t[-1]


# =======================================================================
# SECCION 6 - MONTE CARLO (propagacion de incertidumbre)
# =======================================================================
N_MC = 500
t_eval_mc = np.linspace(0, T_END, 200)  # malla mas gruesa para no saturar memoria/tiempo

mc_P_dCas9_20h = []
mc_P_BirA_20h = []
mc_P_dCas9_traj = []
mc_P_BirA_traj = []
n_exitosas = 0
n_fallidas = 0

print(f"\nCorriendo Monte Carlo (N={N_MC})...")
for i in range(N_MC):
    params_i = params_muestreados_dict()
    sol_i = solve_ivp(
        fun=sistema_odes, t_span=(0, T_END), y0=Y0, method="Radau",
        t_eval=t_eval_mc, args=(params_i,), rtol=1e-7, atol=1e-9,
    )
    if sol_i.success:
        n_exitosas += 1
        P_dCas9_i = np.maximum(sol_i.y[3], 0.0)
        P_BirA_i = np.maximum(sol_i.y[4], 0.0)
        mc_P_dCas9_20h.append(P_dCas9_i[-1])
        mc_P_BirA_20h.append(P_BirA_i[-1])
        mc_P_dCas9_traj.append(P_dCas9_i)
        mc_P_BirA_traj.append(P_BirA_i)
    else:
        n_fallidas += 1

mc_P_dCas9_20h = np.array(mc_P_dCas9_20h)
mc_P_BirA_20h = np.array(mc_P_BirA_20h)
mc_P_dCas9_traj = np.array(mc_P_dCas9_traj)   # shape (n_exitosas, len(t_eval_mc))
mc_P_BirA_traj = np.array(mc_P_BirA_traj)

print(f"Monte Carlo completo: {n_exitosas} corridas exitosas, {n_fallidas} fallidas.")

# Estadisticos resumen (mediana, IC90% = percentiles 5 y 95)
P_dCas9_mediana = np.median(mc_P_dCas9_20h)
P_dCas9_p5 = np.percentile(mc_P_dCas9_20h, 5)
P_dCas9_p95 = np.percentile(mc_P_dCas9_20h, 95)

P_BirA_mediana = np.median(mc_P_BirA_20h)
P_BirA_p5 = np.percentile(mc_P_BirA_20h, 5)
P_BirA_p95 = np.percentile(mc_P_BirA_20h, 95)

frac_BirA_suficiente = np.mean(mc_P_BirA_20h >= mc_P_dCas9_20h) * 100.0

# Bandas de IC90% a lo largo del tiempo (para la Figura 2)
P_dCas9_traj_mediana = np.median(mc_P_dCas9_traj, axis=0)
P_dCas9_traj_p5 = np.percentile(mc_P_dCas9_traj, 5, axis=0)
P_dCas9_traj_p95 = np.percentile(mc_P_dCas9_traj, 95, axis=0)

P_BirA_traj_mediana = np.median(mc_P_BirA_traj, axis=0)
P_BirA_traj_p5 = np.percentile(mc_P_BirA_traj, 5, axis=0)
P_BirA_traj_p95 = np.percentile(mc_P_BirA_traj, 95, axis=0)


# =======================================================================
# SECCION 7 - ANALISIS DE SENSIBILIDAD
# =======================================================================
# --- Sensibilidad 1: f_rt ---
valores_f_rt = [0.05, 0.10, 0.20, 0.30, 0.50]
P_BirA_20h_vs_frt = []
for f_rt_i in valores_f_rt:
    p_i = construir_params_16C(
        IPTG_conc.value, K_IPTG.value, n_Hill.value, k_R_ref.value, Q10_tx.value,
        delta_R_ref.value, Q10_delta_R.value, k_tx_ref.value, delta_m_ref.value,
        Q10_delta_m.value, k_tl0_ref.value, RBS_str_1.value, RBS_str_2.value,
        Q10_tl.value, delta_p_dCas9_ref.value, delta_p_BirA_ref.value, Q10_delta_p.value,
        mu_ref.value, Q10_mu.value, f_rt_i,
    )
    sol_i = solve_ivp(sistema_odes, (0, T_END), Y0, method="Radau",
                       t_eval=[T_END], args=(p_i,), rtol=1e-8, atol=1e-10)
    P_BirA_20h_vs_frt.append(max(sol_i.y[4][-1], 0.0))

# --- Sensibilidad 2: k_R_ref ---
valores_k_R = [5.0, 10.0, 20.0, 35.0, 50.0]
P_dCas9_20h_vs_kR = []
P_BirA_20h_vs_kR = []
for k_R_i in valores_k_R:
    p_i = construir_params_16C(
        IPTG_conc.value, K_IPTG.value, n_Hill.value, k_R_i, Q10_tx.value,
        delta_R_ref.value, Q10_delta_R.value, k_tx_ref.value, delta_m_ref.value,
        Q10_delta_m.value, k_tl0_ref.value, RBS_str_1.value, RBS_str_2.value,
        Q10_tl.value, delta_p_dCas9_ref.value, delta_p_BirA_ref.value, Q10_delta_p.value,
        mu_ref.value, Q10_mu.value, f_rt.value,
    )
    sol_i = solve_ivp(sistema_odes, (0, T_END), Y0, method="Radau",
                       t_eval=[T_END], args=(p_i,), rtol=1e-8, atol=1e-10)
    P_dCas9_20h_vs_kR.append(max(sol_i.y[3][-1], 0.0))
    P_BirA_20h_vs_kR.append(max(sol_i.y[4][-1], 0.0))


# =======================================================================
# SECCION 8 - VISUALIZACION (4 figuras)
# =======================================================================

# --- Figura 1: Dinamica determinista (2x3) ---
fig1, axes1 = plt.subplots(2, 3, figsize=(18, 10))
fig1.suptitle("Modulo A v2.0 - Dinamica determinista (16C, 20h)", fontsize=14, fontweight="bold")

axes1[0, 0].plot(t_horas, R_t, color="tab:brown", linewidth=2)
axes1[0, 0].set_xlabel("Tiempo (h)"); axes1[0, 0].set_ylabel("[T7 RNAP] (nM)")
axes1[0, 0].set_title("T7 RNAP - vista completa"); axes1[0, 0].grid(alpha=0.3)

axes1[0, 1].plot(t_horas, T_L_t, color="tab:orange", linewidth=2, label="T_L")
axes1[0, 1].plot(t_horas, T_2_t, color="tab:cyan", linewidth=2, label="T_2")
axes1[0, 1].set_xlabel("Tiempo (h)"); axes1[0, 1].set_ylabel("Concentracion (nM)")
axes1[0, 1].set_title("Transcritos - vista completa"); axes1[0, 1].legend(fontsize=9); axes1[0, 1].grid(alpha=0.3)

axes1[0, 2].plot(t_horas, P_dCas9_t, color="tab:green", linewidth=2, label="P_dCas9-AviTag")
axes1[0, 2].plot(t_horas, P_BirA_t, color="tab:purple", linewidth=2, label="P_BirA")
axes1[0, 2].set_xlabel("Tiempo (h)"); axes1[0, 2].set_ylabel("Concentracion (nM)")
axes1[0, 2].set_title("Proteinas - vista completa"); axes1[0, 2].legend(fontsize=9); axes1[0, 2].grid(alpha=0.3)

mascara_zoom = t_horas <= 2.0
axes1[1, 0].plot(t_horas[mascara_zoom], R_t[mascara_zoom], color="tab:brown", linewidth=2)
axes1[1, 0].set_xlabel("Tiempo (h)"); axes1[1, 0].set_ylabel("[T7 RNAP] (nM)")
axes1[1, 0].set_title("ZOOM T7 RNAP: primeras 2h"); axes1[1, 0].grid(alpha=0.3)

axes1[1, 1].plot(t_horas[mascara_zoom], T_L_t[mascara_zoom], color="tab:orange", linewidth=2, label="T_L")
axes1[1, 1].plot(t_horas[mascara_zoom], T_2_t[mascara_zoom], color="tab:cyan", linewidth=2, label="T_2")
axes1[1, 1].set_xlabel("Tiempo (h)"); axes1[1, 1].set_ylabel("Concentracion (nM)")
axes1[1, 1].set_title("ZOOM transcritos: primeras 2h"); axes1[1, 1].legend(fontsize=9); axes1[1, 1].grid(alpha=0.3)

axes1[1, 2].plot(t_horas[mascara_zoom], P_dCas9_t[mascara_zoom], color="tab:green", linewidth=2, label="P_dCas9-AviTag")
axes1[1, 2].plot(t_horas[mascara_zoom], P_BirA_t[mascara_zoom], color="tab:purple", linewidth=2, label="P_BirA")
axes1[1, 2].set_xlabel("Tiempo (h)"); axes1[1, 2].set_ylabel("Concentracion (nM)")
axes1[1, 2].set_title("ZOOM proteinas: primeras 2h"); axes1[1, 2].legend(fontsize=9); axes1[1, 2].grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("ModuloA_v2_dinamica_completa.png", dpi=150)
print("\nFigura guardada: ModuloA_v2_dinamica_completa.png")

# --- Figura 2: Outputs para integracion, con banda IC90% Monte Carlo ---
fig2, (ax_dcas9, ax_bira) = plt.subplots(1, 2, figsize=(14, 6))
fig2.suptitle("Modulo A v2.0 - Outputs para integracion (con incertidumbre Monte Carlo)", fontsize=14, fontweight="bold")

P_dCas9_funcional_t = P_dCas9_t * phi_16.value
ax_dcas9.fill_between(t_eval_mc, P_dCas9_traj_p5, P_dCas9_traj_p95, color="tab:green", alpha=0.2, label="IC90% Monte Carlo")
ax_dcas9.plot(t_horas, P_dCas9_t, color="tab:green", linewidth=2, linestyle="-", label="P_dCas9 total (nominal)")
ax_dcas9.plot(t_horas, P_dCas9_funcional_t, color="tab:green", linewidth=2, linestyle=":", label=f"P_dCas9 x phi_16={phi_16.value}")
ax_dcas9.axvline(20.0, color="gray", linestyle="--", linewidth=1)
ax_dcas9.set_xlabel("Tiempo (h)"); ax_dcas9.set_ylabel("Concentracion (nM)")
ax_dcas9.set_title("dCas9-AviTag: total vs. funcional, con incertidumbre")
ax_dcas9.legend(loc="upper left", fontsize=8); ax_dcas9.grid(alpha=0.3)

ax_bira.fill_between(t_eval_mc, P_BirA_traj_p5, P_BirA_traj_p95, color="tab:purple", alpha=0.2, label="IC90% Monte Carlo")
ax_bira.plot(t_horas, P_BirA_t, color="tab:purple", linewidth=2, label="P_BirA (nominal)")
ax_bira.axvline(20.0, color="gray", linestyle="--", linewidth=1)
ax_bira.annotate(f"P_BirA(20h) = {P_BirA_20h_det:.2f} nM\n[IC90%: {P_BirA_p5:.2f}-{P_BirA_p95:.2f}]",
                  xy=(20.0, P_BirA_20h_det), xytext=(11.0, P_BirA_20h_det * 0.6),
                  fontsize=8.5, color="tab:purple", arrowprops=dict(arrowstyle="->", color="tab:purple"))
ax_bira.set_xlabel("Tiempo (h)"); ax_bira.set_ylabel("Concentracion (nM)")
ax_bira.set_title("BirA total, con incertidumbre")
ax_bira.legend(loc="upper left", fontsize=8); ax_bira.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("ModuloA_v2_outputs_integracion.png", dpi=150)
print("Figura guardada: ModuloA_v2_outputs_integracion.png")

# --- Figura 3: Sensibilidad a f_rt y k_R ---
fig3, (ax_frt, ax_kr) = plt.subplots(1, 2, figsize=(14, 6))
fig3.suptitle("Modulo A v2.0 - Analisis de sensibilidad", fontsize=14, fontweight="bold")

ax_frt.plot(valores_f_rt, P_BirA_20h_vs_frt, marker="o", markersize=8, color="tab:purple", linewidth=2, label="P_BirA(20h)")
ax_frt.axhline(P_dCas9_20h_det, color="tab:green", linestyle="--", linewidth=1.5, label=f"P_dCas9(20h) = {P_dCas9_20h_det:.2f} nM")
ax_frt.set_xlabel("f_rt"); ax_frt.set_ylabel("P_BirA(20h) (nM)")
ax_frt.set_title("Sensibilidad a f_rt"); ax_frt.legend(fontsize=9); ax_frt.grid(alpha=0.3)

ax_kr.plot(valores_k_R, P_dCas9_20h_vs_kR, marker="o", markersize=8, color="tab:green", linewidth=2, label="P_dCas9(20h)")
ax_kr.plot(valores_k_R, P_BirA_20h_vs_kR, marker="s", markersize=8, color="tab:purple", linewidth=2, label="P_BirA(20h)")
ax_kr.axvline(k_R_ref.value, color="gray", linestyle=":", linewidth=1.5, label=f"Valor nominal (k_R={k_R_ref.value})")
ax_kr.set_xlabel("k_R_ref (nM/h)"); ax_kr.set_ylabel("Concentracion a 20h (nM)")
ax_kr.set_title("Sensibilidad a k_R_ref"); ax_kr.legend(fontsize=9); ax_kr.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("ModuloA_v2_sensibilidad.png", dpi=150)
print("Figura guardada: ModuloA_v2_sensibilidad.png")

# --- Figura 4: Histogramas Monte Carlo ---
fig4, (ax_h1, ax_h2) = plt.subplots(1, 2, figsize=(14, 6))
fig4.suptitle(f"Distribucion Monte Carlo (N={n_exitosas}) - Modulo A v2.0", fontsize=14, fontweight="bold")

ax_h1.hist(mc_P_dCas9_20h, bins=30, color="tab:green", alpha=0.7, edgecolor="black")
ax_h1.axvline(P_dCas9_mediana, color="black", linewidth=2, label=f"Mediana = {P_dCas9_mediana:.2f} nM")
ax_h1.axvline(P_dCas9_p5, color="black", linestyle="--", linewidth=1, label=f"P5 = {P_dCas9_p5:.2f} nM")
ax_h1.axvline(P_dCas9_p95, color="black", linestyle="--", linewidth=1, label=f"P95 = {P_dCas9_p95:.2f} nM")
ax_h1.set_xlabel("P_dCas9(20h) (nM)"); ax_h1.set_ylabel("Frecuencia")
ax_h1.set_title("Distribucion de P_dCas9(20h)"); ax_h1.legend(fontsize=8); ax_h1.grid(alpha=0.3)

ax_h2.hist(mc_P_BirA_20h, bins=30, color="tab:purple", alpha=0.7, edgecolor="black")
ax_h2.axvline(P_BirA_mediana, color="black", linewidth=2, label=f"Mediana = {P_BirA_mediana:.2f} nM")
ax_h2.axvline(P_BirA_p5, color="black", linestyle="--", linewidth=1, label=f"P5 = {P_BirA_p5:.2f} nM")
ax_h2.axvline(P_BirA_p95, color="black", linestyle="--", linewidth=1, label=f"P95 = {P_BirA_p95:.2f} nM")
ax_h2.set_xlabel("P_BirA(20h) (nM)"); ax_h2.set_ylabel("Frecuencia")
ax_h2.set_title("Distribucion de P_BirA(20h)"); ax_h2.legend(fontsize=8); ax_h2.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("ModuloA_v2_montecarlo.png", dpi=150)
print("Figura guardada: ModuloA_v2_montecarlo.png")

plt.show()


# =======================================================================
# SECCION 9 - OUTPUT DE CONSOLA
# =======================================================================
print("\n" + "=" * 90)
print("1. TABLA DE TODOS LOS PARAMETROS")
print("=" * 90)

# Parametros con correccion Q10 explicita (constantes cineticas de temperatura)
params_con_q10 = [
    ("k_R_ref", k_R_ref, Q10_tx),
    ("delta_R_ref", delta_R_ref, Q10_delta_R),
    ("k_tx_ref", k_tx_ref, Q10_tx),
    ("delta_m_ref", delta_m_ref, Q10_delta_m),
    ("k_tl0_ref", k_tl0_ref, Q10_tl),
    ("delta_p_dCas9_ref", delta_p_dCas9_ref, Q10_delta_p),
    ("delta_p_BirA_ref", delta_p_BirA_ref, Q10_delta_p),
    ("mu_ref", mu_ref, Q10_mu),
]

print(f"{'Nombre':<20s}{'Valor(37C)':>12s}{'Valor(16C)':>12s}{'Q10':>8s}{'Origin':>14s}{'rel_unc':>10s}")
print("-" * 90)
for nombre, param_obj, q10_obj in params_con_q10:
    v16 = ajustar_Q10(param_obj.value, q10_obj.value, T_exp.value, T_ref.value)
    print(f"{nombre:<20s}{param_obj.value:>12.4f}{v16:>12.4f}{q10_obj.value:>8.2f}{param_obj.origin:>14s}{param_obj.rel_unc:>9.0%}")

# Parametros de Q10 en si mismos, y parametros sin escalado por temperatura
print()
print(f"{'Nombre':<20s}{'Valor':>12s}{'Unidad':>14s}{'Origin':>14s}{'rel_unc':>10s}")
print("-" * 90)
otros = ["T_exp", "T_ref", "IPTG_conc", "K_IPTG", "n_Hill", "Q10_tx", "Q10_delta_R",
         "Q10_delta_m", "Q10_tl", "Q10_delta_p", "Q10_mu", "RBS_str_1", "RBS_str_2",
         "f_rt", "phi_16", "phi_37"]
for nombre in otros:
    param_obj = TODOS_LOS_PARAMS[nombre]
    print(f"{nombre:<20s}{param_obj.value:>12.4g}{param_obj.unit:>14s}{param_obj.origin:>14s}{param_obj.rel_unc:>9.0%}")

print("\n" + "=" * 90)
print("2. PARAMETROS WETLAB_PEND / PLACEHOLDER (con nota de que medir)")
print("=" * 90)
for nombre, param_obj in TODOS_LOS_PARAMS.items():
    if param_obj.origin in ("WETLAB_PEND", "PLACEHOLDER") and param_obj.wet_lab_note:
        print(f"\n[{param_obj.origin}] {nombre} = {param_obj.value} {param_obj.unit}")
        print(f"  -> {param_obj.wet_lab_note}")

print("\n" + "=" * 90)
print("3. RESULTADOS DETERMINISTICOS (corrida nominal)")
print("=" * 90)
print(f"P_dCas9(20h)                 = {P_dCas9_20h_det:.3f} nM  -> Modulo C (total)")
print(f"P_dCas9(20h) x phi_16        = {P_dCas9_20h_det * phi_16.value:.3f} nM  -> Modulo C (funcional)")
print(f"P_BirA(20h)                  = {P_BirA_20h_det:.3f} nM  -> Modulo B")
print(f"Relacion P_dCas9 : P_BirA    = 1 : {P_BirA_20h_det / P_dCas9_20h_det:.3f}")

dydt_final = sistema_odes(T_END, [R_t[-1], T_L_t[-1], T_2_t[-1], P_dCas9_t[-1], P_BirA_t[-1]], params_nom)
dP_dCas9_dt_serie = np.gradient(P_dCas9_t, t_horas)
pct_deriv_final = abs(dydt_final[3]) / np.max(np.abs(dP_dCas9_dt_serie)) * 100.0
estado_txt = "EN PLATEAU" if pct_deriv_final < 5.0 else "AUN ACUMULANDO"
print(f"Estado de P_dCas9 a t=20h    = {estado_txt} (dP/dt al {pct_deriv_final:.1f}% del maximo)")

print("\n" + "=" * 90)
print("4. RESULTADOS MONTE CARLO (N={} exitosas)".format(n_exitosas))
print("=" * 90)
print(f"P_dCas9(20h) = {P_dCas9_mediana:.2f} nM  [IC90%: {P_dCas9_p5:.2f} - {P_dCas9_p95:.2f} nM]")
print(f"P_BirA(20h)  = {P_BirA_mediana:.2f} nM  [IC90%: {P_BirA_p5:.2f} - {P_BirA_p95:.2f} nM]")
print(f"% de corridas donde P_BirA(20h) >= P_dCas9(20h): {frac_BirA_suficiente:.1f}%")

print("\n" + "=" * 90)
print("5. ADVERTENCIA DE ESTEQUIOMETRIA")
print("=" * 90)
if P_BirA_20h_det < P_dCas9_20h_det:
    print(f"ADVERTENCIA: en la corrida determinista nominal, P_BirA(20h)={P_BirA_20h_det:.2f} nM "
          f"es MENOR que P_dCas9(20h)={P_dCas9_20h_det:.2f} nM. Esto podria dejar dCas9-AviTag "
          f"sin biotinilar. El Monte Carlo indica que esto ocurre en el {100-frac_BirA_suficiente:.1f}% "
          f"de las combinaciones de parametros muestreadas.")
else:
    print(f"P_BirA(20h)={P_BirA_20h_det:.2f} nM >= P_dCas9(20h)={P_dCas9_20h_det:.2f} nM en la "
          f"corrida nominal: en principio hay suficiente BirA. El Monte Carlo confirma esto en "
          f"el {frac_BirA_suficiente:.1f}% de las combinaciones muestreadas.")

print("\n" + "=" * 90)
print("6. PARAMETROS QUE REQUIEREN MEDICION EXPERIMENTAL (por prioridad)")
print("=" * 90)
prioridad = [
    ("CRITICA", "f_rt", f_rt),
    ("CRITICA", "k_R_ref", k_R_ref),
    ("ALTA", "phi_16", phi_16),
    ("ALTA", "phi_37", phi_37),
    ("MEDIA", "RBS_str_1", RBS_str_1),
    ("MEDIA", "RBS_str_2", RBS_str_2),
    ("MEDIA", "K_IPTG", K_IPTG),
    ("MEDIA", "delta_p_BirA_ref", delta_p_BirA_ref),
    ("BAJA", "mu_ref", mu_ref),
]
for nivel, nombre, param_obj in prioridad:
    print(f"  [{nivel:<8s}] {nombre:<20s} nota: {param_obj.wet_lab_note or '(sin nota especifica)'}")

print("=" * 90)