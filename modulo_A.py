#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=======================================================================
MODULO A - PRODUCCION EN E. coli (SISTEMA DE 5 ODEs, T7 RNAP, 16C / 20h)
=======================================================================
Proyecto iGEM 2025 - Deteccion de mutaciones oncogenicas en ctDNA
Modelado matematico de biologia sintetica (Dry Lab)

Este es el MODULO RAIZ del modelo completo. Sus outputs alimentan a
todos los demas modulos:
    P_dCas9(20h) x phi(16C) -> Modulo C (proteina disponible para purificacion)
    P_BirA(20h)              -> Modulo B (enzima disponible para biotinilacion)
    P_dCas9(20h) : P_BirA(20h) -> resultado de diseno QbD (estequiometria)

# ============================================================
# INSTALACION DE DEPENDENCIAS (ejecutar en terminal una sola vez)
# ============================================================
# conda activate pythonProjectya
# conda install numpy scipy matplotlib
#
# pip (alternativa):
# pip install numpy scipy matplotlib
#
# Verificar:
# python -c "import numpy, scipy, matplotlib; print('OK')"
#
# Ejecutar:
# python modulo_A_expresion_genica.py
# ============================================================
"""

# =======================================================================
# IMPORTS
# =======================================================================
import numpy as np                              # arreglos numericos y operaciones vectoriales
from scipy.integrate import solve_ivp            # resolvedor de sistemas de ODEs (metodo Radau para sistemas stiff)
import matplotlib.pyplot as plt                  # generacion de graficas
import matplotlib.gridspec as gridspec           # layout de subplots mas flexible que plt.subplots

# =======================================================================
# JUSTIFICACION DEL MODELO
# =======================================================================
# ¿POR QUE ODEs AQUI Y NO EN LOS OTROS MODULOS?
# -----------------------------------------------------------------------
# Los Modulos D y E pudieron resolverse con ecuaciones algebraicas de
# equilibrio (isotermas de Langmuir/Hill) porque las reacciones de union
# involucradas (biotina-streptavidina, dCas9-sgRNA-target) alcanzan su
# estado estacionario mucho mas rapido que la escala de tiempo del
# experimento, y solo hay 1-2 especies relevantes.
#
# El Modulo A es distinto: aqui CINCO especies moleculares cambian
# SIMULTANEAMENTE y a velocidades comparables durante las 20h de
# induccion: la T7 RNAP se sintetiza y se degrada, dos tipos de ARNm se
# transcriben y degradan, y dos proteinas se traducen y degradan, todo
# mientras la celula misma crece y diluye su contenido. Ninguna de estas
# especies alcanza equilibrio instantaneo respecto a las demas, por lo
# que el sistema NO puede reducirse a una formula algebraica: se
# requiere resolver el sistema acoplado de ecuaciones diferenciales en
# el tiempo.
#
# DISENO GENETICO DEL SISTEMA
# -----------------------------------------------------------------------
# La construccion tiene DOS promotores T7 en TANDEM y UN terminador
# compartido:
#
#   [Promotor T7 #1] -> [6xHis-TrxA-dCas9-AviTag] -> [Promotor T7 #2] -> [BirA] -> [Terminador T7]
#
# La T7 RNAP que inicia en el Promotor #1 no siempre termina al llegar
# al final del gen dCas9-AviTag: sin terminador intermedio, una fraccion
# f_rt ("read-through") continua transcribiendo hasta el terminador
# compartido, generando un transcrito LARGO (T_L) que contiene AMBAS
# ORFs (dCas9-AviTag y BirA). La T7 RNAP que inicia en el Promotor #2
# genera un transcrito CORTO (T_2) que solo contiene la ORF de BirA.
#
# Consecuencia: BirA se traduce desde DOS fuentes de ARNm (T_L, con
# eficiencia parcial via RBS2, y T_2, con eficiencia completa via RBS2),
# mientras que dCas9-AviTag SOLO se traduce desde T_L (via RBS1). Este
# diseno amplifica deliberadamente la produccion de BirA respecto a
# dCas9-AviTag, para asegurar suficiente enzima disponible para
# biotinilacion completa (ver Modulo B).
#
# VARIABLES DE ESTADO
# -----------------------------------------------------------------------
#   R(t)       : T7 RNAP activa [nM]. Se acumula gradualmente tras la
#                induccion con IPTG (sistema DE3: lacUV5 -> gen de T7 RNAP).
#   T_L(t)     : transcrito largo [nM]. Iniciado en Promotor T7 #1;
#                contiene dCas9-AviTag siempre, y BirA solo si hubo
#                read-through.
#   T_2(t)     : transcrito corto [nM]. Iniciado en Promotor T7 #2;
#                contiene solo BirA.
#   P_dCas9(t) : proteina dCas9-AviTag total [nM]. Traducida solo desde T_L.
#   P_BirA(t)  : proteina BirA total [nM]. Traducida desde T_L (read-through,
#                fraccion f_rt) y desde T_2 (promotor propio).
#
# SIGNIFICADO DE CADA TERMINO EN CADA ODE
# -----------------------------------------------------------------------
# Cada ecuacion tiene la forma general:
#     d[X]/dt = SINTESIS - DEGRADACION - DILUCION
#
# La DILUCION por crecimiento celular (termino mu(T)*X) aplica a TODAS
# las especies: a medida que la celula crece y se divide, su contenido
# molecular se reparte entre las celulas hijas, lo cual equivale
# matematicamente a una perdida de concentracion proporcional a la tasa
# de crecimiento mu. A 16C esta tasa es baja (division cada ~5-6h) pero
# NO despreciable en una induccion de 20h (~3-4 generaciones).
#
# La DEGRADACION (terminos delta_R, delta_m, delta_p) representa la
# destruccion activa de cada especie por maquinaria celular (proteasas
# para proteinas, RNasas para ARNm).
#
# La SINTESIS de cada especie depende de la especie "aguas arriba" en la
# cascada de expresion genica: T7 RNAP se sintetiza a tasa constante tras
# induccion (k_R); los transcritos se sintetizan proporcionalmente a la
# cantidad de T7 RNAP disponible (k_tx1*R, k_tx2*R); las proteinas se
# sintetizan proporcionalmente a la cantidad de su ARNm molde (k_tl1*T_L,
# k_tl2*(f_rt*T_L + T_2)).
#
# ¿POR QUE f_rt ES EL PARAMETRO MAS IMPORTANTE?
# -----------------------------------------------------------------------
# f_rt controla directamente cuanta "ayuda extra" recibe BirA del
# transcrito largo T_L (que existe en la misma cantidad que el ARNm de
# dCas9-AviTag). Es el parametro de diseno genetico mas incierto
# (depende de la eficiencia real de terminacion transcripcional del
# sistema, que no tiene terminador intermedio) y el que mas control tiene
# el equipo sobre la relacion estequiometrica final P_dCas9:P_BirA, que
# es el resultado de diseno (QbD) mas relevante de todo este modulo: si
# f_rt es demasiado bajo, BirA podria ser insuficiente para biotinilar
# todo el dCas9-AviTag producido (cuello de botella para el Modulo B).


# =======================================================================
# SUPUESTOS DEL MODELO
# =======================================================================
# 1. IPTG se anade a t=0 y la induccion es instantanea: no se modela un
#    retraso (lag) en la senalizacion IPTG -> lacUV5 -> T7 RNAP.
# 2. k_tx1 = k_tx2 = k_tx_ref: se asume que los dos promotores T7 en
#    tandem tienen la misma fuerza (no hay medicion de fuerza relativa
#    individual). Supuesto documentado, a refinar con WetLab.
# 3. Los dos transcritos (T_L, T_2) comparten la misma tasa de
#    degradacion delta_m (ambos son ARNm con estructura 5'/3' similar,
#    sin datos que sugieran degradacion diferencial).
# 4. dCas9-AviTag y BirA comparten la misma tasa de degradacion de
#    proteina delta_p (ambas se asumen proteinas estables tipicas de
#    E. coli, sin datos de vida media especifica para cada una).
# 5. El factor de plegamiento phi(T) NO forma parte de las ODEs: se
#    aplica como factor multiplicativo EXTERNO al output final de
#    P_dCas9(20h), porque describe una propiedad de la POBLACION de
#    proteina ya sintetizada (que fraccion plego correctamente) y no un
#    proceso cinetico adicional de sintesis/degradacion en el tiempo.
# 6. Sistema cerrado: no hay exportacion o secrecion de proteina fuera
#    de la celula (dCas9-AviTag y BirA son intracelulares).
# 7. Temperatura constante (16C) durante toda la induccion de 20h (no se
#    modela el tiempo de enfriamiento del cultivo tras la induccion).
# =======================================================================


# =======================================================================
# FUNCION DE CORRECCION DE TEMPERATURA (Q10)
# =======================================================================
def ajustar_Q10(k_ref, Q10, T_exp=16.0, T_ref=37.0):
    """
    Ajusta una constante cinetica medida a T_ref hacia una nueva
    temperatura T_exp utilizando el factor Q10.

        k_ajustado = k_ref * Q10 ^ ((T_exp - T_ref) / 10)

    Parametros:
        k_ref  : constante cinetica a la temperatura de referencia
        Q10    : factor de sensibilidad termica del proceso (adimensional)
        T_exp  : temperatura experimental (C)
        T_ref  : temperatura de referencia de la literatura (C)

    Retorna:
        k_ajustado : constante cinetica corregida a T_exp
    """
    return k_ref * (Q10 ** ((T_exp - T_ref) / 10.0))


# =======================================================================
# PARAMETROS DEL MODELO
# =======================================================================
# Convencion de comentarios:
#   [unidad] | Fuente: cita APA 7
#   ⚠ WET LAB: indica que el valor debe reemplazarse con dato experimental
#   ← Modulo X: indica que el valor proviene del output de otro modulo
#   → Modulo X: indica que el valor calculado aqui alimenta a otro modulo

# ================================================================
# TEMPERATURAS
# ================================================================
T_exp = 16.0   # [C] | Temperatura de induccion experimental del proyecto iGEM 2025.
               #        Estrategia de cold shock para mejorar plegamiento de dCas9.
               #        Fuente: Vera, A., Gonzalez-Montalban, N., Aris, A., &
               #        Villaverde, A. (2007). The conformational quality of
               #        insoluble recombinant proteins is enhanced at low growth
               #        temperatures. Biotechnology and Bioengineering, 96(6), 1101-1106.
               #        https://doi.org/10.1002/bit.21218

T_ref = 37.0   # [C] | Temperatura de referencia para parametros de literatura.
               #        La mayoria de los parametros de E. coli estan reportados a 37C.

# ================================================================
# TASA DE CRECIMIENTO DE E. coli
# ================================================================
mu_ref = 1.386  # [h^-1] | Tasa de crecimiento a 37C en medio LB.
                #           Equivalente a tiempo de duplicacion ~30 min.
                #           Fuente: Farewell, A., & Neidhardt, F. C. (1998). Effect of
                #           temperature on in vivo protein synthetic capacity in
                #           Escherichia coli. Journal of Bacteriology, 180(17), 4704-4710.
                #           https://doi.org/10.1128/JB.180.17.4704-4710.1998

Q10_mu = 2.5    # [adim] | Coeficiente Q10 para tasa de crecimiento de E. coli.
                #           El crecimiento es muy sensible a temperatura (Q10 alto).
                #           Fuente: Farewell & Neidhardt (1998). Journal of Bacteriology,
                #           180(17), 4704-4710. https://doi.org/10.1128/JB.180.17.4704-4710.1998
                # ⚠ WET LAB: medir curva de crecimiento a 600 nm a 16C post-induccion
                #            para obtener mu(16C) real de su cultivo.

# ================================================================
# T7 RNAP - SINTESIS Y DEGRADACION
# ================================================================
k_R_ref = 5.0   # [nM/h] | Tasa de sintesis de T7 RNAP bajo induccion IPTG en BL21(DE3).
                #           Valor representativo post-induccion a saturacion.
                #           Fuente: Carrier, T. A., & Keasling, J. D. (1999). Library of
                #           synthetic 5' secondary structures to manipulate mRNA stability
                #           in Escherichia coli. Biotechnology Progress, 15(1), 58-64.
                #           https://doi.org/10.1021/bp9801143
                # ⚠ WET LAB: valor de referencia; depende de la concentracion de IPTG
                #            y cepa utilizada.

Q10_tx = 1.8    # [adim] | Q10 para actividad de T7 RNAP (sintesis de RNAP activa Y
                #           tasa de transcripcion por molecula de RNAP). Se usa para
                #           ajustar tanto k_R como k_tx (ver nota de diseno abajo).
                #           T7 RNAP es termoestable pero su actividad baja a 16C.
                #           Fuente: Chamberlin, M., & Ring, J. (1973). Characterization
                #           of T7-specific ribonucleic acid polymerase. Journal of
                #           Biological Chemistry, 248(6), 2235-2244.
                #           https://doi.org/10.1016/S0021-9258(19)44178-4
                # NOTA DE DISENO: el documento fuente agrupa Q10_tx junto a k_R_ref,
                #           pero el nombre ("tx" = transcripcion) tambien coincide con
                #           k_tx_ref. Ante esta ambiguedad, se decidio aplicar el MISMO
                #           Q10_tx a k_R_ref y a k_tx_ref, ya que ambos procesos dependen
                #           de la actividad de la maquinaria de T7 RNAP. Esto es una
                #           decision de modelado documentada, no un dato de literatura
                #           adicional; WetLab puede desacoplarlos si obtiene datos mas
                #           granulares en el futuro.

delta_R_ref = 0.2  # [h^-1] | Tasa de degradacion de T7 RNAP a 37C.
                   #           Vida media ~3.5h. Proteina relativamente estable.
                   #           Fuente: Perez-Perez, J., & Gutierrez, J. (1995).
                   #           An arabinose-inducible expression vector, pAR3, compatible
                   #           with ColE1-derived plasmids. Gene, 158(1), 141-142.
                   #           https://doi.org/10.1016/0378-1119(95)00127-R

Q10_delta_R = 1.5  # [adim] | Q10 para degradacion de proteinas estables (T7 RNAP).
                   #           Fuente: Goldberg, A. L. (2003). Protein degradation and
                   #           protection against misfolded or damaged proteins. Nature,
                   #           426(6968), 895-899. https://doi.org/10.1038/nature02263

# ================================================================
# TRANSCRIPCION - TRANSCRITOS T_L Y T_2
# ================================================================
k_tx_ref = 2.0   # [h^-1 nM^-1] | Tasa de transcripcion por molecula de T7 RNAP.
                 #                 Valor de referencia para promotor T7 fuerte (PT7).
                 #                 Se asume k_tx1 = k_tx2 = k_tx_ref (promotores T7 en
                 #                 tandem de fuerza equivalente). Supuesto documentado.
                 #                 Fuente: Golding, I., Paulsson, J., Zawilski, S. M., &
                 #                 Cox, E. C. (2005). Real-time kinetics of gene activity
                 #                 in individual bacteria. Cell, 123(6), 1025-1036.
                 #                 https://doi.org/10.1016/j.cell.2005.09.031

delta_m_ref = 6.0  # [h^-1] | Tasa de degradacion de ARNm en E. coli a 37C.
                   #           Vida media de ARNm: ~5-8 min a 37C -> delta_m ~6-8 h^-1.
                   #           Fuente: Bernstein, J. A., Khodursky, A. B., Lin, P. H.,
                   #           Lin-Chao, S., & Cohen, S. N. (2002). Global analysis of
                   #           mRNA decay and abundance in Escherichia coli at single-gene
                   #           resolution using two-color fluorescent DNA microarrays.
                   #           Proceedings of the National Academy of Sciences, 99(15),
                   #           9697-9702. https://doi.org/10.1073/pnas.112318199

Q10_delta_m = 2.0  # [adim] | Q10 para degradacion de ARNm.
                   #           La degradacion de ARNm es enzimatica (RNasas) y sensible
                   #           a temperatura. A 16C, los ARNm son mas estables.
                   #           Fuente: Phadtare, S., & Severinov, K. (2010). RNA remodeling
                   #           and gene regulation by cold shock proteins. RNA Biology,
                   #           7(6), 788-795. https://doi.org/10.4161/rna.7.6.13482

# ================================================================
# TRADUCCION - dCas9-AviTag (RBS1) Y BirA (RBS2)
# ================================================================
k_tl1_ref = 3.0  # [h^-1 nM^-1] | Tasa de traduccion RBS1 (dCas9-AviTag) a 37C.
                 #                 Valor representativo para RBS de fuerza media-alta.
                 #                 Fuente: Salis, H. M., Mirsky, E. A., & Voigt, C. A.
                 #                 (2009). Automated design of synthetic ribosome binding
                 #                 sites to control protein expression. Nature
                 #                 Biotechnology, 27(10), 946-950.
                 #                 https://doi.org/10.1038/nbt.1568
                 # ⚠ WET LAB: usar RBS Calculator (Salis Lab, https://salislab.net/software/)
                 #            con la secuencia real de RBS1 para obtener Translation
                 #            Initiation Rate (TIR) especifico de su construccion.

k_tl2_ref = 2.0  # [h^-1 nM^-1] | Tasa de traduccion RBS2 (BirA) a 37C.
                 #                 Se asume RBS2 ligeramente menos fuerte que RBS1.
                 #                 Fuente: Salis et al. (2009). Nature Biotechnology,
                 #                 27(10), 946-950. https://doi.org/10.1038/nbt.1568
                 # ⚠ WET LAB: usar RBS Calculator con la secuencia real de RBS2.

Q10_tl = 2.2     # [adim] | Q10 para traduccion (actividad ribosomal).
                 #           Los ribosomas son muy sensibles a temperatura baja.
                 #           Fuente: Broeze, R. J., Solomon, C. J., & Pope, D. H. (1978).
                 #           Effects of low temperature on in vivo and in vitro protein
                 #           synthesis in Escherichia coli and Pseudomonas fluorescens.
                 #           Journal of Bacteriology, 134(3), 861-874.
                 #           https://doi.org/10.1128/jb.134.3.861-874.1978

# ================================================================
# DEGRADACION DE PROTEINAS
# ================================================================
delta_p_ref = 0.05  # [h^-1] | Tasa de degradacion de proteinas estables a 37C.
                    #           Vida media ~14h. dCas9 y BirA son proteinas estables.
                    #           Fuente: Maurizi, M. R. (1992). Proteases and protein
                    #           degradation in Escherichia coli. Experientia, 48(2),
                    #           178-201. https://doi.org/10.1007/BF01923511

Q10_delta_p = 1.5   # [adim] | Q10 para degradacion de proteinas (proteasas).
                    #           Fuente: Goldberg, A. L. (2003). Nature, 426(6968), 895-899.
                    #           https://doi.org/10.1038/nature02263

# ================================================================
# PARAMETROS DE DISENO GENETICO
# ================================================================
f_rt = 0.2   # [adim] | Eficiencia de read-through del Promotor T7 #1 hacia BirA.
             #           Fraccion de T7 RNAP que inicia en Promotor #1 y continua
             #           transcribiendo hasta incluir la ORF de BirA.
             #           Rango plausible en sistemas T7 sin terminador intermedio: 0.1-0.5.
             #           Fuente: Masulis, I. S., et al. (2015). Efficiency of
             #           transcription termination in E. coli T7 systems.
             #           Molecular Microbiology, 96(5), 962-978. Valor placeholder.
             # ⚠ WET LAB: medir por RT-qPCR comparando abundancia relativa de la
             #            region 3' de dCas9 vs. region de BirA en el ARNm total.
             #            Este es el parametro mas incierto del modulo.
             #            NOTA: se explora f_rt = 0.05, 0.10, 0.20, 0.30, 0.50 en el
             #            analisis de sensibilidad (Figura 3).

# ================================================================
# FACTOR DE PLEGAMIENTO - PUENTE CON MODULO C
# ================================================================
phi_16 = 0.60  # [adim] | Fraccion de dCas9-AviTag correctamente plegada a 16C.
               #           La induccion en frio reduce la formacion de cuerpos de
               #           inclusion en proteinas grandes y de plegamiento dificil.
               #           Rango esperado a 16C: 0.4-0.8.
               #           Fuente: Vera et al. (2007). Biotechnology and
               #           Bioengineering, 96(6), 1101-1106.
               #           https://doi.org/10.1002/bit.21218
               # ⚠ WET LAB PRIORITARIO: medir fraccion soluble vs. insoluble por
               #            SDS-PAGE tras lisis a 16C y 37C. Este es el dato
               #            experimental de mayor impacto en el modelo completo.
               # → Modulo C: multiplica a P_dCas9(20h) para dar la proteina
               #            funcional disponible para purificacion.

phi_37 = 0.15  # [adim] | Fraccion de dCas9-AviTag plegada a 37C (referencia, no se
               #           usa en la simulacion principal a 16C, se deja documentada
               #           para comparacion futura).
               #           Alta tendencia a inclusion a temperatura estandar.
               #           Fuente: Kusano, S., Ding, Q., Fujita, N., & Ishihama, A.
               #           (1993). Functional differences between SigmaD and SigmaS
               #           Escherichia coli RNA polymerase. Molecular Microbiology,
               #           10(3), 575-584. Estimado general.
               # ⚠ WET LAB: confirmar con experimento comparativo de induccion.

# ================================================================
# CONDICIONES INICIALES Y TIEMPO DE SIMULACION
# ================================================================
R0        = 0.0   # [nM] | T7 RNAP inicial = 0 (induccion empieza en t=0)
T_L0      = 0.0   # [nM] | Transcrito largo inicial = 0
T_20      = 0.0   # [nM] | Transcrito corto inicial = 0
P_dCas9_0 = 0.0   # [nM] | dCas9-AviTag inicial = 0
P_BirA_0  = 0.0   # [nM] | BirA inicial = 0

t_start = 0.0    # [h] | Inicio de la induccion
t_end   = 20.0   # [h] | Tiempo de induccion experimental


# =======================================================================
# AJUSTE DE TODAS LAS CONSTANTES CINETICAS A 16C (Q10)
# =======================================================================
k_R_adj       = ajustar_Q10(k_R_ref, Q10_tx, T_exp, T_ref)
delta_R_adj   = ajustar_Q10(delta_R_ref, Q10_delta_R, T_exp, T_ref)
mu_adj        = ajustar_Q10(mu_ref, Q10_mu, T_exp, T_ref)
k_tx1_adj     = ajustar_Q10(k_tx_ref, Q10_tx, T_exp, T_ref)   # promotor T7 #1
k_tx2_adj     = ajustar_Q10(k_tx_ref, Q10_tx, T_exp, T_ref)   # promotor T7 #2 (= k_tx1_adj, supuesto)
delta_m_adj   = ajustar_Q10(delta_m_ref, Q10_delta_m, T_exp, T_ref)
k_tl1_adj     = ajustar_Q10(k_tl1_ref, Q10_tl, T_exp, T_ref)
k_tl2_adj     = ajustar_Q10(k_tl2_ref, Q10_tl, T_exp, T_ref)
delta_p_adj   = ajustar_Q10(delta_p_ref, Q10_delta_p, T_exp, T_ref)


def imprimir_tabla_parametros():
    """
    Imprime una tabla comparativa de todas las constantes cineticas del
    sistema a 37C (valor de literatura) y a 16C (valor ajustado por Q10),
    junto con el factor de cambio resultante.
    """
    filas = [
        ("k_R (sintesis T7 RNAP)",     "nM/h",       k_R_ref,      k_R_adj,     Q10_tx),
        ("delta_R (degrad. T7 RNAP)",  "h^-1",       delta_R_ref,  delta_R_adj, Q10_delta_R),
        ("mu (crecimiento)",           "h^-1",       mu_ref,       mu_adj,      Q10_mu),
        ("k_tx1 = k_tx2 (transcr.)",   "h^-1 nM^-1", k_tx_ref,     k_tx1_adj,   Q10_tx),
        ("delta_m (degrad. ARNm)",     "h^-1",       delta_m_ref,  delta_m_adj, Q10_delta_m),
        ("k_tl1 (traducc. RBS1)",      "h^-1 nM^-1", k_tl1_ref,    k_tl1_adj,   Q10_tl),
        ("k_tl2 (traducc. RBS2)",      "h^-1 nM^-1", k_tl2_ref,    k_tl2_adj,   Q10_tl),
        ("delta_p (degrad. proteina)", "h^-1",       delta_p_ref,  delta_p_adj, Q10_delta_p),
    ]
    print("-" * 90)
    print(f"{'Parametro':<28s}{'Unidad':<13s}{'Valor 37C':>12s}{'Valor 16C':>12s}{'Q10':>8s}{'Factor cambio':>16s}")
    print("-" * 90)
    for nombre, unidad, v37, v16, q10 in filas:
        factor_cambio = v16 / v37 if v37 != 0 else float("nan")
        print(f"{nombre:<28s}{unidad:<13s}{v37:>12.4f}{v16:>12.4f}{q10:>8.2f}{factor_cambio:>16.3f}")
    print("-" * 90)


print("=" * 90)
print("MODULO A - Produccion de dCas9-AviTag y BirA en E. coli (T7 RNAP, 16C, 20h)")
print("=" * 90)
print("Tabla comparativa de parametros cineticos: 37C (literatura) vs 16C (ajustado por Q10)")
imprimir_tabla_parametros()


# =======================================================================
# SISTEMA DE 5 ODEs
# =======================================================================
def sistema_odes(t, y, params):
    """
    Sistema de 5 ecuaciones diferenciales acopladas que describe la
    produccion de T7 RNAP, dos transcritos (T_L, T_2) y dos proteinas
    (dCas9-AviTag, BirA) durante la induccion a 16C.

    Variables de estado (vector y):
        y[0] = R        -> [T7 RNAP activa] (nM)
        y[1] = T_L      -> [transcrito largo] (nM)
        y[2] = T_2      -> [transcrito corto] (nM)
        y[3] = P_dCas9  -> [dCas9-AviTag total] (nM)
        y[4] = P_BirA   -> [BirA total] (nM)
    """
    # Proteccion contra valores negativos por error numerico del integrador
    R = max(y[0], 0.0)
    T_L = max(y[1], 0.0)
    T_2 = max(y[2], 0.0)
    P_dCas9 = max(y[3], 0.0)
    P_BirA = max(y[4], 0.0)

    k_R = params["k_R"]
    delta_R = params["delta_R"]
    mu = params["mu"]
    k_tx1 = params["k_tx1"]
    k_tx2 = params["k_tx2"]
    delta_m = params["delta_m"]
    k_tl1 = params["k_tl1"]
    k_tl2 = params["k_tl2"]
    delta_p = params["delta_p"]
    f_rt_local = params["f_rt"]

    # --- ODE 1: T7 RNAP ---
    # Sintesis constante post-induccion (k_R) menos degradacion propia
    # (delta_R*R) menos dilucion por crecimiento celular (mu*R).
    dR_dt = k_R - (delta_R + mu) * R

    # --- ODE 2: Transcrito largo T_L (dCas9-AviTag + BirA por read-through) ---
    # Sintesis proporcional a la T7 RNAP disponible (k_tx1*R) menos
    # degradacion de ARNm (delta_m*T_L) menos dilucion (mu*T_L).
    dT_L_dt = k_tx1 * R - (delta_m + mu) * T_L

    # --- ODE 3: Transcrito corto T_2 (solo BirA) ---
    # Misma logica que T_L, pero desde el Promotor T7 #2.
    dT_2_dt = k_tx2 * R - (delta_m + mu) * T_2

    # --- ODE 4: Proteina dCas9-AviTag ---
    # Traduccion solo desde T_L (via RBS1) menos degradacion de proteina
    # (delta_p*P_dCas9) menos dilucion (mu*P_dCas9).
    dP_dCas9_dt = k_tl1 * T_L - (delta_p + mu) * P_dCas9

    # --- ODE 5: Proteina BirA ---
    # Traduccion desde DOS fuentes: el read-through del transcrito largo
    # (f_rt*T_L) y el transcrito corto propio (T_2), ambas via RBS2;
    # menos degradacion y dilucion.
    dP_BirA_dt = k_tl2 * (f_rt_local * T_L + T_2) - (delta_p + mu) * P_BirA

    return [dR_dt, dT_L_dt, dT_2_dt, dP_dCas9_dt, dP_BirA_dt]


# =======================================================================
# RESOLUCION NUMERICA
# =======================================================================
parametros_odes = {
    "k_R": k_R_adj,
    "delta_R": delta_R_adj,
    "mu": mu_adj,
    "k_tx1": k_tx1_adj,
    "k_tx2": k_tx2_adj,
    "delta_m": delta_m_adj,
    "k_tl1": k_tl1_adj,
    "k_tl2": k_tl2_adj,
    "delta_p": delta_p_adj,
    "f_rt": f_rt,
}

y0 = [R0, T_L0, T_20, P_dCas9_0, P_BirA_0]
t_eval = np.linspace(t_start, t_end, 5000)   # alta resolucion en horas

sol = solve_ivp(
    fun=sistema_odes,
    t_span=(t_start, t_end),
    y0=y0,
    method="Radau",   # metodo robusto para sistemas stiff (ver Seccion 12 al final)
    t_eval=t_eval,
    dense_output=True,
    args=(parametros_odes,),
    rtol=1e-8,
    atol=1e-10,
)

if sol.success:
    print("\nIntegracion numerica: EXITOSA (Radau convergio correctamente)")
else:
    print("\nADVERTENCIA: la integracion numerica NO convergio.")
    print(f"Mensaje del solver: {sol.message}")

# Extraccion de las 5 series de tiempo, con clip de seguridad >= 0
t_horas = sol.t
R_t = np.maximum(sol.y[0], 0.0)
T_L_t = np.maximum(sol.y[1], 0.0)
T_2_t = np.maximum(sol.y[2], 0.0)
P_dCas9_t = np.maximum(sol.y[3], 0.0)
P_BirA_t = np.maximum(sol.y[4], 0.0)


# =======================================================================
# VISUALIZACION - FIGURA 1: DINAMICA COMPLETA (2x3)
# =======================================================================
fig1 = plt.figure(figsize=(18, 10))
gs1 = gridspec.GridSpec(2, 3, figure=fig1)
fig1.suptitle("Modulo A - Dinamica completa de produccion (16C, 20h)", fontsize=14, fontweight="bold")

ax_R = fig1.add_subplot(gs1[0, 0])
ax_T = fig1.add_subplot(gs1[0, 1])
ax_P = fig1.add_subplot(gs1[0, 2])
ax_R_zoom = fig1.add_subplot(gs1[1, 0])
ax_T_zoom = fig1.add_subplot(gs1[1, 1])
ax_P_zoom = fig1.add_subplot(gs1[1, 2])

# --- Panel (1,1): R(t) vista completa ---
ax_R.plot(t_horas, R_t, color="tab:brown", linewidth=2)
ax_R.set_xlabel("Tiempo (h)")
ax_R.set_ylabel("[T7 RNAP] (nM)")
ax_R.set_title("(1,1) T7 RNAP - vista completa (0-20h)")
ax_R.grid(alpha=0.3)

# --- Panel (1,2): T_L(t) y T_2(t) vista completa ---
ax_T.plot(t_horas, T_L_t, color="tab:orange", linewidth=2, label="T_L (transcrito largo)")
ax_T.plot(t_horas, T_2_t, color="tab:cyan", linewidth=2, label="T_2 (transcrito corto)")
ax_T.set_xlabel("Tiempo (h)")
ax_T.set_ylabel("Concentracion (nM)")
ax_T.set_title("(1,2) Transcritos - vista completa (0-20h)")
ax_T.legend(loc="best", fontsize=9)
ax_T.grid(alpha=0.3)

# --- Panel (1,3): P_dCas9(t) y P_BirA(t) vista completa ---
ax_P.plot(t_horas, P_dCas9_t, color="tab:green", linewidth=2, label="P_dCas9-AviTag")
ax_P.plot(t_horas, P_BirA_t, color="tab:purple", linewidth=2, label="P_BirA")
ax_P.set_xlabel("Tiempo (h)")
ax_P.set_ylabel("Concentracion (nM)")
ax_P.set_title("(1,3) Proteinas - vista completa (0-20h)")
ax_P.legend(loc="best", fontsize=9)
ax_P.grid(alpha=0.3)

# --- Datos de zoom (primeras 2h), usando la solucion densa para alta resolucion ---
t_zoom = np.linspace(0.0, 2.0, 500)
estado_zoom = sol.sol(t_zoom)
R_zoom = np.maximum(estado_zoom[0], 0.0)
T_L_zoom = np.maximum(estado_zoom[1], 0.0)
T_2_zoom = np.maximum(estado_zoom[2], 0.0)
P_dCas9_zoom = np.maximum(estado_zoom[3], 0.0)
P_BirA_zoom = np.maximum(estado_zoom[4], 0.0)

# --- Panel (2,1): ZOOM T7 RNAP, primeras 2h ---
ax_R_zoom.plot(t_zoom, R_zoom, color="tab:brown", linewidth=2)
ax_R_zoom.set_xlabel("Tiempo (h)")
ax_R_zoom.set_ylabel("[T7 RNAP] (nM)")
ax_R_zoom.set_title("(2,1) ZOOM T7 RNAP: primeras 2h")
ax_R_zoom.grid(alpha=0.3)

# --- Panel (2,2): ZOOM transcritos, primeras 2h ---
ax_T_zoom.plot(t_zoom, T_L_zoom, color="tab:orange", linewidth=2, label="T_L")
ax_T_zoom.plot(t_zoom, T_2_zoom, color="tab:cyan", linewidth=2, label="T_2")
ax_T_zoom.set_xlabel("Tiempo (h)")
ax_T_zoom.set_ylabel("Concentracion (nM)")
ax_T_zoom.set_title("(2,2) ZOOM transcritos: primeras 2h")
ax_T_zoom.legend(loc="best", fontsize=9)
ax_T_zoom.grid(alpha=0.3)

# --- Panel (2,3): ZOOM proteinas, primeras 2h ---
ax_P_zoom.plot(t_zoom, P_dCas9_zoom, color="tab:green", linewidth=2, label="P_dCas9-AviTag")
ax_P_zoom.plot(t_zoom, P_BirA_zoom, color="tab:purple", linewidth=2, label="P_BirA")
ax_P_zoom.set_xlabel("Tiempo (h)")
ax_P_zoom.set_ylabel("Concentracion (nM)")
ax_P_zoom.set_title("(2,3) ZOOM proteinas: primeras 2h")
ax_P_zoom.legend(loc="best", fontsize=9)
ax_P_zoom.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("ModuloA_dinamica_completa.png", dpi=150)
print("\nFigura guardada como: ModuloA_dinamica_completa.png")


# =======================================================================
# VISUALIZACION - FIGURA 2: OUTPUTS PARA INTEGRACION
# =======================================================================
fig2, (ax_dcas9, ax_bira) = plt.subplots(1, 2, figsize=(14, 6))
fig2.suptitle("Modulo A - Outputs para integracion con Modulos B y C", fontsize=14, fontweight="bold")

# --- Panel izquierdo: P_dCas9 total vs P_dCas9 x phi(16C) ---
P_dCas9_funcional_t = P_dCas9_t * phi_16
ax_dcas9.plot(t_horas, P_dCas9_t, color="tab:green", linewidth=2, linestyle="-", label="P_dCas9-AviTag total")
ax_dcas9.plot(t_horas, P_dCas9_funcional_t, color="tab:green", linewidth=2, linestyle=":", label=f"P_dCas9-AviTag x phi(16C={phi_16})")
ax_dcas9.axvline(20.0, color="gray", linestyle="--", linewidth=1)
ax_dcas9.text(20.0, ax_dcas9.get_ylim()[1] * 0.05, " Output -> Modulo C", rotation=90, va="bottom", ha="right", fontsize=8, color="gray")
ax_dcas9.set_xlabel("Tiempo (h)")
ax_dcas9.set_ylabel("Concentracion (nM)")
ax_dcas9.set_title("dCas9-AviTag: total vs. fraccion funcional")
ax_dcas9.legend(loc="upper left", fontsize=9)
ax_dcas9.grid(alpha=0.3)

# --- Panel derecho: P_BirA(t) ---
P_BirA_20h = P_BirA_t[-1]
ax_bira.plot(t_horas, P_BirA_t, color="tab:purple", linewidth=2, label="P_BirA")
ax_bira.axvline(20.0, color="gray", linestyle="--", linewidth=1)
ax_bira.text(20.0, P_BirA_20h * 0.05, " Output -> Modulo B", rotation=90, va="bottom", ha="right", fontsize=8, color="gray")
ax_bira.annotate(f"P_BirA(20h) = {P_BirA_20h:.2f} nM",
                  xy=(20.0, P_BirA_20h), xytext=(12.0, P_BirA_20h * 0.85),
                  fontsize=9, color="tab:purple",
                  arrowprops=dict(arrowstyle="->", color="tab:purple", lw=1))
ax_bira.set_xlabel("Tiempo (h)")
ax_bira.set_ylabel("Concentracion (nM)")
ax_bira.set_title("BirA total")
ax_bira.legend(loc="upper left", fontsize=9)
ax_bira.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("ModuloA_outputs_integracion.png", dpi=150)
print("Figura guardada como: ModuloA_outputs_integracion.png")


# =======================================================================
# VISUALIZACION - FIGURA 3: SENSIBILIDAD DE P_BirA(20h) A f_rt
# =======================================================================
valores_f_rt = [0.05, 0.10, 0.20, 0.30, 0.50]
P_BirA_20h_por_frt = []

for f_rt_i in valores_f_rt:
    parametros_i = dict(parametros_odes)   # copia de los parametros base
    parametros_i["f_rt"] = f_rt_i
    sol_i = solve_ivp(
        fun=sistema_odes,
        t_span=(t_start, t_end),
        y0=y0,
        method="Radau",
        t_eval=[t_end],   # solo se necesita el valor final en t=20h
        args=(parametros_i,),
        rtol=1e-8,
        atol=1e-10,
    )
    P_BirA_20h_i = max(sol_i.y[4][-1], 0.0)
    P_BirA_20h_por_frt.append(P_BirA_20h_i)

P_dCas9_20h = P_dCas9_t[-1]   # referencia 1:1 para la Figura 3

fig3, ax_sens = plt.subplots(1, 1, figsize=(9, 6.5))
ax_sens.plot(valores_f_rt, P_BirA_20h_por_frt, marker="o", markersize=8, color="tab:purple",
             linewidth=2, label="P_BirA(20h) simulado")
ax_sens.axhline(P_dCas9_20h, color="tab:green", linestyle="--", linewidth=1.5,
                 label=f"P_dCas9(20h) = {P_dCas9_20h:.2f} nM (referencia 1:1)")
ax_sens.set_xlabel("f_rt (fraccion de read-through)")
ax_sens.set_ylabel("P_BirA(20h) (nM)")
ax_sens.set_title("Sensibilidad de P_BirA(20h) al parametro de read-through f_rt")
ax_sens.legend(loc="best", fontsize=9)
ax_sens.grid(alpha=0.3)
# Este analisis justifica el valor de f_rt elegido: se busca el minimo f_rt
# que garantiza P_BirA(20h) >= P_dCas9(20h), condicion necesaria (aunque no
# suficiente) para biotinilacion completa de todo el dCas9-AviTag producido.

plt.tight_layout()
plt.savefig("ModuloA_sensibilidad_frt.png", dpi=150)
print("Figura guardada como: ModuloA_sensibilidad_frt.png")

plt.show()


# =======================================================================
# OUTPUT EN CONSOLA
# =======================================================================
print("\n" + "=" * 90)
print("OUTPUT PARA INTEGRACION - MODULO A")
print("=" * 90)

# (a) Tabla comparativa de parametros (repetida aqui para tenerla en el resumen final)
print("\n(a) Tabla comparativa de parametros 37C vs 16C:")
imprimir_tabla_parametros()

# (b) y (c) P_dCas9(20h) y fraccion funcional
P_dCas9_20h = P_dCas9_t[-1]
P_dCas9_funcional_20h = P_dCas9_20h * phi_16
print(f"\n(b) P_dCas9(20h): {P_dCas9_20h:.3f} nM  # -> Output para Modulo C (proteina total)")
print(f"(c) P_dCas9(20h) x phi(16C={phi_16}): {P_dCas9_funcional_20h:.3f} nM  "
      f"# -> Output para Modulo C (proteina funcional plegada)")

# (d) P_BirA(20h)
P_BirA_20h = P_BirA_t[-1]
print(f"(d) P_BirA(20h): {P_BirA_20h:.3f} nM  # -> Output para Modulo B (BirA_0)")

# (e) Relacion estequiometrica
relacion_dcas9_bira = P_dCas9_20h / P_BirA_20h if P_BirA_20h > 0 else float("inf")
print(f"(e) Relacion P_dCas9(20h) : P_BirA(20h) = 1 : {P_BirA_20h / P_dCas9_20h:.2f}  "
      f"(equivalente a P_dCas9/P_BirA = {relacion_dcas9_bira:.3f})")


def tiempo_para_pct_maximo(pct_objetivo, t_arr, y_arr):
    """
    Interpola el tiempo (en horas) en el que una serie y_arr alcanza un
    porcentaje objetivo de su valor MAXIMO dentro del rango simulado.
    """
    y_max = np.max(y_arr)
    if y_max <= 0:
        return None
    objetivo = pct_objetivo * y_max
    if y_arr[-1] < objetivo:
        return None
    return np.interp(objetivo, y_arr, t_arr)


# (f) Tiempo para 50% y 90% del valor maximo de cada proteina
print("\n(f) Tiempo para alcanzar 50% y 90% del valor maximo de cada proteina:")
for nombre_especie, y_arr in [("P_dCas9-AviTag", P_dCas9_t), ("P_BirA", P_BirA_t)]:
    t_50 = tiempo_para_pct_maximo(0.50, t_horas, y_arr)
    t_90 = tiempo_para_pct_maximo(0.90, t_horas, y_arr)
    t_50_str = f"{t_50:.2f} h" if t_50 is not None else "no alcanzado"
    t_90_str = f"{t_90:.2f} h" if t_90 is not None else "no alcanzado"
    print(f"    {nombre_especie:<16s} 50%: {t_50_str:<14s} 90%: {t_90_str}")

# (g) Estado del sistema a t=20h: ¿plateau o aun acumulando?
print("\n(g) Estado del sistema a t=20h (comparando dP/dt en t=20h vs el maximo dP/dt observado):")
derivadas_finales = sistema_odes(t_end, [R_t[-1], T_L_t[-1], T_2_t[-1], P_dCas9_t[-1], P_BirA_t[-1]], parametros_odes)
nombres_especies = ["R (T7 RNAP)", "T_L", "T_2", "P_dCas9-AviTag", "P_BirA"]
series_especies = [R_t, T_L_t, T_2_t, P_dCas9_t, P_BirA_t]

advertencias_plateau = []
for nombre_especie, y_arr, dydt_final in zip(nombres_especies, series_especies, derivadas_finales):
    dydt_serie = np.gradient(y_arr, t_horas)
    dydt_max = np.max(np.abs(dydt_serie))
    if dydt_max > 0:
        pct_derivada_final = abs(dydt_final) / dydt_max * 100.0
    else:
        pct_derivada_final = 0.0
    estado = "EN PLATEAU (< 5% del dP/dt maximo)" if pct_derivada_final < 5.0 else "AUN ACUMULANDO"
    print(f"    {nombre_especie:<16s} dP/dt(20h) = {pct_derivada_final:5.1f}% del maximo -> {estado}")

    # Verificacion adicional: ¿la especie llego a 95% de su valor final antes de t=10h?
    idx_10h = np.argmin(np.abs(t_horas - 10.0))
    valor_10h = y_arr[idx_10h]
    valor_final = y_arr[-1]
    if valor_final > 0 and (valor_10h / valor_final) > 0.95:
        advertencias_plateau.append(nombre_especie)

if advertencias_plateau:
    print(f"\n    ADVERTENCIA: las siguientes especies alcanzaron >95% de su valor final "
          f"de t=20h ANTES de t=10h: {', '.join(advertencias_plateau)}.")
    print("    Esto sugiere que, para esas especies, la degradacion y/o dilucion son")
    print("    relativamente rapidas frente a la sintesis a 16C, alcanzando el estado")
    print("    estacionario bastante antes del final de la induccion. Si esto no es lo")
    print("    esperado biologicamente, revisar los valores de Q10 de degradacion/sintesis")
    print("    correspondientes.")
else:
    print("\n    Ninguna especie parece haber alcanzado plateau antes de t=10h.")

# (h) Advertencia si P_BirA(20h) < P_dCas9(20h)
print()
if P_BirA_20h < P_dCas9_20h:
    print(f"(h) ADVERTENCIA: P_BirA(20h) = {P_BirA_20h:.3f} nM es MENOR que "
          f"P_dCas9(20h) = {P_dCas9_20h:.3f} nM. Esto significa que, en el peor caso "
          f"(1 BirA por 1 dCas9-AviTag), podria no haber suficiente BirA para "
          f"biotinilar todo el dCas9-AviTag producido. Considerar aumentar f_rt o la "
          f"fuerza de RBS2.")
else:
    print(f"(h) P_BirA(20h) = {P_BirA_20h:.3f} nM >= P_dCas9(20h) = {P_dCas9_20h:.3f} nM: "
          f"hay suficiente BirA (en base molar) para, en principio, biotinilar todo "
          f"el dCas9-AviTag producido.")

# (i) Comentario sobre el impacto de f_rt
print(f"\n(i) Con f_rt = {f_rt}, el transcrito largo T_L contribuye "
      f"{f_rt*100:.0f}% de su concentracion a la traduccion de BirA (ademas del 100% "
      f"del transcrito corto T_2). El analisis de sensibilidad (Figura 3) muestra como "
      f"cambia P_BirA(20h) al variar f_rt entre 0.05 y 0.50, y permite identificar el "
      f"valor minimo de f_rt necesario para mantener P_BirA(20h) >= P_dCas9(20h).")

print("=" * 90)


# =======================================================================
# SECCION DE INTEGRACION CON MODULOS SIGUIENTES
# =======================================================================
#
#   1. P_dCas9(20h) x phi(16C) (calculado arriba) se pasa al Modulo C como
#      P_total_disponible: la cantidad de proteina dCas9-AviTag
#      correctamente plegada, disponible para el proceso de purificacion.
#
#   2. P_BirA(20h) se pasa al Modulo B como BirA_0, REEMPLAZANDO el valor
#      provisional de 1 uM (1000 nM) que se uso en la version standalone
#      de ese modulo. Nota de unidades: Modulo B trabaja en M, este
#      modulo trabaja en nM; recordar convertir (1 nM = 1e-9 M) al pasar
#      el valor entre modulos.
#
#   3. Cuando WetLab mida phi(16C) real (SDS-PAGE de fraccion soluble),
#      el UNICO cambio necesario es actualizar el parametro phi_16 en
#      este script; el resto del pipeline se recalcula automaticamente.
#
#   4. Cuando WetLab mida f_rt real (RT-qPCR comparando abundancia de
#      dCas9 vs. BirA en el ARNm), el UNICO cambio necesario es actualizar
#      el parametro f_rt en este script.
#
#   5. La relacion P_dCas9(20h) : P_BirA(20h) es un CPP (Critical Process
#      Parameter) del sistema QbD: cuantifica si el diseno genetico
#      (dos promotores T7 + read-through) esta logrando su objetivo de
#      amplificar la produccion de BirA respecto a dCas9-AviTag.
#
# =======================================================================


# =======================================================================
# NOTAS SOBRE DISENO Y ROBUSTEZ NUMERICA
# =======================================================================
#
# ¿POR QUE SE USO EL METODO RADAU EN LUGAR DE RK45?
# -----------------------------------------------------------------------
# Este sistema de 5 ODEs es "stiff" (rigido): las escalas de tiempo de
# las distintas especies difieren en varios ordenes de magnitud dentro
# del mismo sistema. Por ejemplo, la degradacion de ARNm (delta_m, del
# orden de horas^-1 incluso a 16C) es mucho mas rapida que la dilucion
# por crecimiento celular (mu, mucho mas lenta a 16C). Un integrador
# explicito como RK45 necesitaria pasos de tiempo extremadamente
# pequenos para mantenerse estable frente al proceso mas rapido del
# sistema, incluso cuando se esta simulando la evolucion del proceso mas
# lento — esto lo vuelve computacionalmente ineficiente o inestable.
#
# Radau es un metodo IMPLICITO (de la familia Runge-Kutta), disenado
# especificamente para sistemas stiff: puede tomar pasos de tiempo mucho
# mas grandes sin perder estabilidad numerica, porque resuelve un
# sistema de ecuaciones (generalmente no lineal) en cada paso en lugar
# de solo evaluar la derivada hacia adelante. Es mas costoso por paso,
# pero mucho mas eficiente en total para este tipo de sistemas.
#
# ¿QUE SIGNIFICA "STIFF" EN ESTE CONTEXTO BIOLOGICO?
# -----------------------------------------------------------------------
# En terminos biologicos, "stiff" refleja que este sistema tiene
# procesos "rapidos" (degradacion de ARNm, con vida media de minutos
# incluso a 16C) acoplados a procesos "lentos" (crecimiento celular y
# acumulacion de proteina estable, en la escala de horas). Matematicamente,
# esto se traduce en autovalores del sistema linealizado que difieren en
# varios ordenes de magnitud, lo cual es la definicion tecnica de
# "stiffness" en ecuaciones diferenciales.
#
# ¿COMO INTERPRETAR SI LAS CURVAS MUESTRAN PLATEAU ANTES DE t=20h?
# -----------------------------------------------------------------------
# El sistema imprime automaticamente una advertencia (seccion "g" del
# output) si alguna especie alcanza mas del 95% de su valor final antes
# de t=10h. Esto NO es necesariamente un error: T7 RNAP y los transcritos
# tienen tiempos de relajacion mas cortos que las proteinas (por el efecto
# de "cascada" en la expresion genica, cada capa aguas abajo tarda mas en
# alcanzar su estado estacionario). Sin embargo, si TODAS las especies
# (incluyendo las proteinas finales) muestran plateau muy temprano, esto
# podria indicar que las tasas de degradacion/dilucion ajustadas a 16C
# son proporcionalmente muy altas frente a las tasas de sintesis, lo cual
# valdria la pena revisar contra los valores de Q10 usados (especialmente
# Q10_delta_m y Q10_delta_p, que determinan que tan rapido decae la
# "memoria" de sintesis previa).
#
# ¿POR QUE EL MODELO NO INCLUYE EL EFECTO DE PLEGAMIENTO EN LAS ODEs?
# -----------------------------------------------------------------------
# El factor de plegamiento phi(T) describe que FRACCION de las moleculas
# de dCas9-AviTag ya sintetizadas terminan en una conformacion
# correctamente plegada y funcional, en oposicion a quedar atrapadas en
# cuerpos de inclusion (agregados insolubles). Este es un proceso que
# ocurre "en paralelo" a la sintesis, determinado principalmente por la
# velocidad de sintesis, la temperatura, y la propia secuencia de la
# proteina — no es una reaccion cinetica adicional de conversion entre
# dos estados moleculares que dependa explicitamente del tiempo de forma
# sencilla de modelar sin datos experimentales detallados de cinetica de
# plegamiento (que no estan disponibles). Por eso se modela como un
# FACTOR EXTERNO fijo que se aplica sobre la salida final P_dCas9(20h)
# en lugar de como una ODE adicional: esto captura el efecto neto
# (fraccion funcional final) sin necesitar asumir una cinetica de
# plegamiento especifica que no esta respaldada por datos propios del
# equipo. Cuando WetLab mida phi(16C) experimentalmente (SDS-PAGE de
# fraccion soluble vs. insoluble), ese valor unico reemplaza el
# parametro y ajusta automaticamente todo el pipeline aguas abajo
# (Modulo C en adelante).
# =======================================================================