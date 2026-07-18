#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=======================================================================
MODULO D - INMOVILIZACION EN CHIP (LANGMUIR, BIOTINA-STREPTAVIDINA)
=======================================================================
Proyecto iGEM 2025 - Modelado matematico de biologia sintetica (Dry Lab)

Este script simula la inmovilizacion de dCas9-Biotin en un chip de
streptavidina, usando dos enfoques complementarios:
    1) Isoterma de Langmuir (equilibrio)
    2) Cinetica de Langmuir (ODE, adsorcion en el tiempo)

Pipeline del proyecto:
    Modulo A -> Modulo B -> Modulo C -> [MODULO D] -> Modulo E -> Modulo F

# ============================================================
# INSTALACION DE DEPENDENCIAS (ejecutar en terminal una sola vez)
# ============================================================
# Si usas conda (recomendado):
#   conda activate iGEM
#   conda install numpy scipy matplotlib
#
# Verificar instalacion:
#   python -c "import numpy, scipy, matplotlib; print('OK')"
#
# Ejecutar el script:
#   python modulo_D_langmuir.py
# ============================================================

=======================================================================
SUPUESTOS DEL MODELO
=======================================================================
1. koff != 0 en realidad (koff=6.8x10^-5 s^-1 para streptavidina silvestre,
   t1/2~2.83 h -- CORREGIDO respecto a una version anterior que asumia
   t1/2~12 dias por una cita equivocada). Sin embargo, durante la
   incubacion ACTIVA (con proteina en exceso en solucion), el termino de
   adsorcion (kon*[P]) domina sobre el termino de desorcion (koff) por
   varios ordenes de magnitud a las concentraciones de trabajo de este
   proyecto, por lo que ignorar la desorcion sigue siendo una aproximacion
   razonable PARA ESTA ETAPA (no necesariamente para un paso posterior de
   lavado/regeneracion del chip, que si dependeria de koff real).
2. Gamma_max se toma de un valor generico de literatura para chips de
   streptavidina. WetLab debe reemplazarlo por el valor real del
   fabricante del chip, o por un experimento de saturacion en SPR.
3. La concentracion de dCas9-Biotin presentada al chip (C_proteina)
   proviene del output del Modulo C (fraccion funcional purificada).
   Aqui se usa un valor provisional de ejemplo.
4. La temperatura no afecta significativamente la union biotina-SA,
   ya que es una interaccion extremadamente estable termicamente.
5. Se ignora la difusion hacia la superficie: se asume un sistema bien
   mezclado (well-mixed), donde [P] es uniforme y constante durante la
   incubacion (no se agota apreciablemente el volumen de muestra).
6. Se asume que toda la proteina presentada al chip ya es biotinilada
   y funcional; el porcentaje de biotinilacion y la fraccion funcional
   ya fueron aplicados en los Modulos B y C respectivamente.
=======================================================================
"""

# =======================================================================
# IMPORTS
# =======================================================================
import numpy as np                                  # arreglos numericos y operaciones vectoriales
from scipy.integrate import solve_ivp                # resolvedor de ODEs para la cinetica de Langmuir
import matplotlib.pyplot as plt                      # generacion de graficas
import matplotlib.ticker as ticker                   # formato de ejes en notacion cientifica (escala log)

# =======================================================================
# PARAMETROS DEL MODELO
# =======================================================================
# Convencion de comentarios:
#   [unidad] | Fuente: cita APA 7
#   ⚠ WET LAB: indica que el valor debe reemplazarse con dato experimental

# --- Constante de asociacion (kon) ---
# CORRECCION: la version anterior citaba a Chivers et al. (2011, Biochemical
# Journal) para este valor, pero ese paper es un estudio de CRISTALOGRAFIA de
# traptavidina (una variante de streptavidina disenada para tener menor koff),
# no reporta directamente un valor de kon para streptavidina silvestre. Se
# corrige la cita a un estudio cinetico real que si mide y discute valores de
# kon en el rango relevante para superficies (no en solucion, donde el kon
# efectivo puede ser mayor por difusion).
kon = 7.5e6  # [M^-1 s^-1] | Fuente: Delgadillo, R. F., Mueser, T. C., Zaleta-Rivera, K.,
             #                Carnes, K. A., Gonzalez-Valdez, J., & Parkhurst, L. J. (2019).
             #                Detailed characterization of the solution kinetics and
             #                thermodynamics of biotin, biocytin and HABA binding to avidin
             #                and streptavidin. PLOS ONE, 14(2), e0204194.
             #                https://doi.org/10.1371/journal.pone.0204194
             #                Este estudio reporta que estudios de union en superficie y de
             #                molecula unica dan kon en el rango 10^5-10^7 M^-1 s^-1 para
             #                streptavidina (mas lento que el limite difusional ~10^9 M^-1 s^-1
             #                que se asumia antes). Se usa 7.5x10^6, dentro de ese rango, como
             #                valor representativo para union EN SUPERFICIE (relevante para un
             #                chip SPR), no para union en solucion libre.
             # ⚠ WET LAB: el rango real reportado en literatura abarca 2 ordenes de
             #             magnitud (10^5-10^7); medir kon especifico del chip/lote usado
             #             por el equipo mejoraria sustancialmente la precision de este valor.

# --- Constante de disociacion cinetica (koff) ---
# CORRECCION IMPORTANTE: la version anterior usaba koff=0.0 justificado con un
# valor citado de ~6x10^-8 s^-1 (t1/2~12 dias) atribuido a Chivers et al. (2010).
# Al verificar la cita, el valor correcto reportado en ese paper (Chivers, C. E.,
# Crozat, E., Chu, C., Moy, V. T., Sherratt, D. J., & Howarth, M. (2010). A
# streptavidin variant with slower biotin dissociation and increased
# mechanostability. Nature Methods, 7(5), 391-393. https://doi.org/10.1038/nmeth.1450)
# para STREPTAVIDINA SILVESTRE (no la variante "traptavidina" disenada en ese
# mismo paper) es koff = 6.8x10^-5 s^-1 a 37C, pH 7.4 -- casi 1000x MAS RAPIDO
# que el valor citado antes. Esto da t1/2 = ln(2)/koff ~ 2.83 horas, no 12 dias.
koff = 6.8e-5  # [s^-1] | Fuente: Chivers, C. E., Crozat, E., Chu, C., Moy, V. T.,
               #           Sherratt, D. J., & Howarth, M. (2010). A streptavidin variant
               #           with slower biotin dissociation and increased mechanostability.
               #           Nature Methods, 7(5), 391-393. https://doi.org/10.1038/nmeth.1450
               #           Valor medido para streptavidina SILVESTRE (no traptavidina) a
               #           37C, pH 7.4. t1/2 = ln(2)/koff ~ 2.83 h.
               #           NOTA IMPORTANTE: aunque koff ya NO es despreciable en escalas de
               #           tiempo largas (horas), en este modelo SI sigue siendo despreciable
               #           frente al termino de adsorcion durante la incubacion activa, porque
               #           kon*[P] >> koff a las concentraciones de trabajo de este proyecto
               #           (ver verificacion numerica impresa en consola mas abajo). La
               #           simplificacion practica (ignorar desorcion durante la incubacion)
               #           se mantiene, pero por la razon correcta -- no porque el complejo
               #           sea casi permanente en la escala de dias.

# --- Constante de disociacion biotina-streptavidina (Kd) ---
# CORRECCION: en vez de tomar un Kd independiente de otra fuente (que generaba
# una inconsistencia interna: Kd, kon y koff deben cumplir Kd = koff/kon), se
# DERIVA Kd a partir de los valores de kon y koff ya verificados arriba, para
# que el modelo sea internamente consistente.
Kd = koff / kon  # [M] | DERIVADO de koff/kon (arriba). Kd ~ 9.07 pM.
                 #        NOTA: este valor (~9x10^-12 M) es mas alto (union mas debil) que
                 #        el Kd ~10^-15 M frecuentemente citado en revisiones generales
                 #        (ej. Green, N. M. (1990). Avidin and streptavidin. Methods in
                 #        Enzymology, 184, 51-67. https://doi.org/10.1016/0076-6879(90)84260-J,
                 #        el valor usado en la version anterior de este script). La diferencia
                 #        refleja que Green (1990) reporta afinidad en SOLUCION/condiciones
                 #        idealizadas, mientras que kon y koff aqui vienen de mediciones mas
                 #        recientes relevantes a union en SUPERFICIE. Se prioriza la
                 #        consistencia interna (Kd=koff/kon con los mismos valores que
                 #        alimentan la ODE cinetica) sobre combinar Kd de una fuente distinta
                 #        a kon/koff. De cualquier forma, Kd~9 pM sigue siendo >1000x menor
                 #        que las concentraciones de trabajo de este proyecto (nM), por lo que
                 #        la conclusion cualitativa (saturacion practicamente completa en
                 #        equilibrio) no cambia.


# --- Capacidad maxima de la superficie (Gamma_max) ---
Gamma_max = 5e-14  # [mol/mm^2] | Valor corregido: equivale a ~50 fmol/mm^2 (5x10^-14 mol/mm^2),
                    #              consistente con ~8,000 RU para una proteina de 160 kDa (dCas9-
                    #              AviTag-TrxA), rango realista para chips SA (streptavidin) de
                    #              Biacore/GE Healthcare.
                    #              CORRECCION: el valor anterior (5x10^-12 mol/mm^2) estaba
                    #              expresado incorrectamente como si fuera del orden de
                    #              pmol/mm^2 (1-5x10^-12 mol/mm^2); la capacidad real de un chip
                    #              SA esta en el orden de fmol/mm^2 (10^-14 - 10^-15 mol/mm^2),
                    #              no de pmol/mm^2.
                    #              NOTA DE VERIFICACION ADICIONAL: Rich & Myszka (2000, Current
                    #              Opinion in Biotechnology) es una revision general de
                    #              metodologia de analisis de biosensores SPR -- NO reporta un
                    #              valor numerico especifico de Gamma_max para chips SA. Se cita
                    #              solo como referencia de contexto/orden de magnitud de la
                    #              sensibilidad tipica de SPR, no como fuente directa de este
                    #              numero. El valor en si sigue siendo un ⚠ WETLAB_PEND genuino.
                    #              Rich, R. L., & Myszka, D. G. (2000). Advances in surface
                    #              plasmon resonance biosensor analysis. Current Opinion in
                    #              Biotechnology, 11(1), 54-61.
                    #              https://doi.org/10.1016/S0958-1669(99)00054-3
                    # ⚠ WET LAB: reemplazar con Gamma_max real del chip especifico del
                    #             fabricante, o con experimento de saturacion en SPR
                    #             cuando este disponible. PRIORITARIO.

# --- Peso molecular para conversion a unidades SPR (RU) ---
# 1 RU (Resonance Unit) equivale aproximadamente a 1 pg/mm^2 de masa inmovilizada
# en la superficie del chip. Esta es la unidad estandar reportada por equipos SPR
# (ej. Biacore) y permite comparar Gamma_final del modelo con datos experimentales.
MW_dCas9 = 160000  # [g/mol] | Peso molecular estimado dCas9-AviTag-TrxA
                    #            Fuente: calculo teorico basado en secuencia de aminoacidos.
                    # ⚠ WET LAB: confirmar por SDS-PAGE o espectrometria de masas.

# --- Concentracion de dCas9-Biotin presentada al chip ---
# ACTUALIZACION: el Modulo C (purificacion) todavia NO esta implementado, asi
# que estos valores vienen DIRECTAMENTE del output del Modulo B (actualizado a
# su vez con el Modulo A v2.0), usados como proxy temporal de C_proteina. Esto
# es una simplificacion: el Modulo C real probablemente incluira un
# rendimiento de purificacion (yield_purif) menor al 100% -- en el modelo de
# Valeria, yield_purif se estimo en ~0.55 -- que reduciria estos numeros aun
# mas. Mientras el Modulo C no exista, se usa el output de B tal cual, y se
# marca explicitamente esta limitacion.
#
# Al igual que en el Modulo B, se usan TRES escenarios (pesimista/nominal/
# optimista) heredados de la incertidumbre Monte Carlo del Modulo A v2.0, en
# vez de un solo valor puntual.
C_proteina = 20.145e-9       # [M] | ← Modulo B, escenario NOMINAL (pct_biotinilacion~100%,
                              #        [dCas9-Biotin]_final del escenario nominal).
                              # ⚠ PENDIENTE: falta Modulo C (purificacion); este valor NO
                              #   incluye perdidas de rendimiento de purificacion todavia.

C_proteina_pesimista = 5.78e-9    # [M] | ← Modulo B, escenario PESIMISTA (P5 de A v2.0)
C_proteina_optimista = 50.64e-9   # [M] | ← Modulo B, escenario OPTIMISTA (P95 de A v2.0)


# --- Tiempo de incubacion en chip ---
t_incubacion = 3600  # [s] | Tiempo tipico de incubacion en chip de streptavidina: 1 hora.
                      #        Fuente: Biosensing Instruments Application Note 123.
                      #        Surface plasmon resonance measurement of protein-peptide
                      #        interaction using streptavidin sensor chip.
                      #        https://biosensingusa.com/application-notes/application-note-123/
                      # ⚠ WET LAB: ajustar segun protocolo experimental real del equipo.


# =======================================================================
# FUNCION: CONVERSION A UNIDADES SPR (RU)
# =======================================================================
def convertir_a_RU(Gamma, MW):
    """
    Convierte una densidad de receptores Gamma [mol/mm^2] a unidades de
    resonancia SPR (RU, Resonance Units).

    Formula de conversion:
        masa_por_area [pg/mm^2] = Gamma [mol/mm^2] * MW [g/mol] * 1e12 [pg/g]
        RU ~ masa_por_area [pg/mm^2]   (1 RU ~ 1 pg/mm^2, convencion estandar SPR)

    Fuente de la convencion 1 RU ~ 1 pg/mm^2:
        Rich, R. L., & Myszka, D. G. (2000). Advances in surface plasmon
        resonance biosensor analysis. Current Opinion in Biotechnology,
        11(1), 54-61. https://doi.org/10.1016/S0958-1669(99)00054-3

    Parametros:
        Gamma : densidad de receptores [mol/mm^2] (puede ser escalar o arreglo)
        MW    : peso molecular de la proteina [g/mol]

    Retorna:
        RU : densidad equivalente en unidades de resonancia SPR
    """
    masa_por_area_pg_mm2 = Gamma * MW * 1e12   # mol/mm^2 * g/mol * pg/g -> pg/mm^2
    RU = masa_por_area_pg_mm2                  # 1 RU ~ 1 pg/mm^2
    return RU


# =======================================================================
# ENFOQUE 1 - ISOTERMA DE LANGMUIR (EQUILIBRIO)
# =======================================================================
def langmuir_equilibrio(P, Gamma_max, Kd):
    """
    Calcula la densidad de receptores ocupados en equilibrio segun la
    isoterma de Langmuir.

        Gamma_eq = Gamma_max * [P] / (Kd + [P])

    Parametros:
        P         : concentracion de proteina presentada al chip [M]
                    (puede ser escalar o arreglo de concentraciones)
        Gamma_max : capacidad maxima de la superficie [mol/mm^2]
        Kd        : constante de disociacion [M]

    Retorna:
        Gamma_eq : densidad de receptores ocupados en equilibrio [mol/mm^2]
    """
    Gamma_eq = Gamma_max * P / (Kd + P)
    return Gamma_eq


# Barrido de concentraciones de 1 fM (1e-18... en este caso 1e-15) a 1e-5 M
# Se usa el rango pedido: 1e-18 M a 1e-5 M, 100 puntos en escala logaritmica
P_barrido = np.logspace(-18, -5, 100)   # [M] concentraciones barridas
Gamma_eq_barrido = langmuir_equilibrio(P_barrido, Gamma_max, Kd)
fraccion_ocupacion_barrido = Gamma_eq_barrido / Gamma_max

# Fraccion de ocupacion a la concentracion especifica que viene del Modulo B (escenario nominal)
Gamma_eq_C = langmuir_equilibrio(C_proteina, Gamma_max, Kd)
fraccion_ocupacion_C = Gamma_eq_C / Gamma_max

print("=" * 70)
print("MODULO D - Inmovilizacion de dCas9-Biotin en chip de streptavidina")
print("=" * 70)
print("--- Enfoque 1: Isoterma de equilibrio de Langmuir ---")
print(f"Concentracion de proteina (Modulo B, escenario nominal): {C_proteina:.2e} M")
print(f"Fraccion de ocupacion en equilibrio a esa concentracion: {fraccion_ocupacion_C*100:.2f}%")

# --- Verificacion numerica: ¿koff es realmente despreciable frente a kon*C? ---
# Esto justifica (o no) la aproximacion de "union practicamente irreversible"
# durante la incubacion activa, ahora que koff ya no es un valor cercano a 0.
tasa_adsorcion = kon * C_proteina
razon_adsorcion_desorcion = tasa_adsorcion / koff
print(f"\nVerificacion koff vs. kon*C_proteina:")
print(f"  kon x C_proteina = {tasa_adsorcion:.4e} s^-1")
print(f"  koff             = {koff:.4e} s^-1")
print(f"  Razon (kon x C_proteina) / koff = {razon_adsorcion_desorcion:.1f}x")
if razon_adsorcion_desorcion > 100:
    print("  -> koff es DESPRECIABLE frente a la adsorcion durante la incubacion activa")
    print("     (la aproximacion de union irreversible es valida PARA ESTA ETAPA).")
else:
    print("  -> ADVERTENCIA: koff podria NO ser despreciable a esta concentracion;")
    print("     revisar si conviene incluir el termino de desorcion explicitamente.")



# =======================================================================
# ENFOQUE 2 - CINETICA DE LANGMUIR (ODE)
# =======================================================================
def modelo_langmuir_cinetico(t, y, kon, koff, C_proteina, Gamma_max):
    """
    Sistema de ODE de un solo estado que describe la adsorcion cinetica
    de dCas9-Biotin sobre la superficie del chip de streptavidina.

    Variable de estado:
        y[0] = Gamma -> densidad de receptores ocupados [mol/mm^2]

    Ecuacion completa (adsorcion + desorcion):
        dGamma/dt = kon * [P] * (Gamma_max - Gamma) - koff * Gamma

    Dado que koff ~ 0 (union practicamente irreversible), el termino de
    desorcion se conserva en la formula por completitud pero su efecto
    es despreciable con koff = 0.0.
    """
    Gamma = max(y[0], 0.0)   # proteccion contra valores negativos por error numerico

    # Termino de adsorcion: proporcional a la concentracion de proteina libre
    # y a los sitios de superficie aun disponibles (Gamma_max - Gamma)
    termino_adsorcion = kon * C_proteina * (Gamma_max - Gamma)

    # Termino de desorcion: proporcional a los receptores ya ocupados
    # (se incluye por completitud del modelo, aunque koff = 0)
    termino_desorcion = koff * Gamma

    dGamma_dt = termino_adsorcion - termino_desorcion
    return [dGamma_dt]


# Condicion inicial: superficie vacia al comenzar la incubacion
Gamma_0 = 0.0   # [mol/mm^2]

# Malla de tiempo para evaluar la solucion (graficas suaves)
t_eval_cinetico = np.linspace(0, t_incubacion, 1000)

sol_cinetica = solve_ivp(
    fun=modelo_langmuir_cinetico,
    t_span=(0, t_incubacion),
    y0=[Gamma_0],
    method="RK45",
    t_eval=t_eval_cinetico,
    dense_output=True,
    args=(kon, koff, C_proteina, Gamma_max),
    # Tolerancias mas estrictas que el default: Gamma trabaja en escala de
    # 1e-14 mol/mm^2 (fmol/mm^2), mucho menor que el atol por defecto de
    # solve_ivp (1e-6), lo que causaria perdida de precision cerca de
    # Gamma=0 o Gamma=Gamma_max.
    rtol=1e-8,
    atol=1e-22,
)

if sol_cinetica.success:
    print("Integracion numerica (cinetica): EXITOSA (RK45 convergio correctamente)")
else:
    print("ADVERTENCIA: la integracion cinetica NO convergio.")
    print(f"Mensaje del solver: {sol_cinetica.message}")

t_seg_cinetico = sol_cinetica.t
Gamma_t = np.maximum(sol_cinetica.y[0], 0.0)   # clip de seguridad >= 0
Gamma_t = np.minimum(Gamma_t, Gamma_max)        # clip de seguridad <= Gamma_max
t_min_cinetico = t_seg_cinetico / 60.0          # conversion a minutos para graficar

fraccion_ocupacion_t = Gamma_t / Gamma_max


def tiempo_para_alcanzar_fraccion(fraccion_objetivo, t_arr, fraccion_arr):
    """
    Interpola el tiempo en el que la curva de ocupacion cruza una
    fraccion objetivo de Gamma_max (ej. 0.5, 0.9, 0.95).
    Retorna None si el objetivo no se alcanza dentro de la simulacion.
    """
    if fraccion_arr[-1] < fraccion_objetivo:
        return None
    t_cruce = np.interp(fraccion_objetivo, fraccion_arr, t_arr)
    return t_cruce


print("--- Enfoque 2: Cinetica de Langmuir (ODE) ---")
for obj in [0.50, 0.90, 0.95]:
    t_cruce_seg = tiempo_para_alcanzar_fraccion(obj, t_seg_cinetico, fraccion_ocupacion_t)
    if t_cruce_seg is None:
        print(f"Tiempo para alcanzar {obj*100:.0f}% de Gamma_max: NO alcanzado en {t_incubacion/60:.0f} min de incubacion")
    else:
        print(f"Tiempo para alcanzar {obj*100:.0f}% de Gamma_max: {t_cruce_seg/60:.2f} min ({t_cruce_seg:.1f} s)")

print("=" * 70)


# =======================================================================
# ENFOQUE 2b - BARRIDO CINETICO PARA MULTIPLES CONCENTRACIONES
# =======================================================================
# Esto responde a una pregunta distinta de la isoterma de equilibrio:
# "¿que tan RAPIDO se llena la superficie segun la concentracion de
# proteina que se use?" Es la informacion clave para elegir la
# concentracion de trabajo en un experimento real de SPR, donde el
# tiempo de inyeccion es limitado.
#
# Concentraciones a comparar: 1 pM, 10 pM, 100 pM, 1 nM, 10 nM, 100 nM
concentraciones_barrido = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7]   # [M]
etiquetas_concentracion = ["1 pM", "10 pM", "100 pM", "1 nM", "10 nM", "100 nM"]

# Cada concentracion tiene una escala de tiempo caracteristica muy distinta
# (tau ~ 1 / (kon * C)). Para poder comparar todas las curvas en la misma
# grafica se usa un tiempo maximo de simulacion largo (24 h) y un eje X en
# escala logaritmica, de forma que tanto las concentraciones rapidas
# (100 nM, saturan en segundos) como las lentas (1 pM, tardan horas) sean
# visibles en el mismo panel.
t_max_barrido = 24 * 3600.0   # [s] 24 horas, suficiente para ver la forma completa incluso a 1 pM
t_eval_barrido = np.logspace(-1, np.log10(t_max_barrido), 300)   # de 0.1 s a 24 h, log-espaciado

resultados_barrido = {}   # guarda fraccion_ocupacion(t) para cada concentracion

print("--- Enfoque 2b: Barrido de concentraciones (cinetica) ---")
for C_i, etiqueta in zip(concentraciones_barrido, etiquetas_concentracion):
    sol_i = solve_ivp(
        fun=modelo_langmuir_cinetico,
        t_span=(0, t_max_barrido),
        y0=[Gamma_0],
        method="RK45",
        t_eval=t_eval_barrido,
        args=(kon, koff, C_i, Gamma_max),
        rtol=1e-8,
        atol=1e-22,
    )
    Gamma_i = np.clip(sol_i.y[0], 0.0, Gamma_max)
    fraccion_i = Gamma_i / Gamma_max
    resultados_barrido[etiqueta] = fraccion_i

    t_90_i = tiempo_para_alcanzar_fraccion(0.90, t_eval_barrido, fraccion_i)
    if t_90_i is None:
        print(f"  [{etiqueta:>7s}] tiempo para 90% de Gamma_max: NO alcanzado en 24 h")
    else:
        print(f"  [{etiqueta:>7s}] tiempo para 90% de Gamma_max: {t_90_i:.2f} s ({t_90_i/60:.2f} min)")

print("=" * 70)


# =======================================================================
# VISUALIZACION
# =======================================================================
fig, axes = plt.subplots(2, 3, figsize=(19, 11))
fig.suptitle("Modulo D - Inmovilizacion de dCas9-Biotin en Chip de Streptavidina", fontsize=14, fontweight="bold")
ax1, ax2, ax3 = axes[0, 0], axes[0, 1], axes[0, 2]
ax4, ax5, ax6 = axes[1, 0], axes[1, 1], axes[1, 2]

# --- (a) Isoterma de Langmuir: fraccion de ocupacion vs [P] (log scale) ---
ax1.plot(P_barrido, fraccion_ocupacion_barrido, color="tab:blue", linewidth=2)
ax1.set_xscale("log")
ax1.axvline(C_proteina, color="tab:red", linestyle="--", linewidth=1.5, label="Input Modulo B (nominal)")
ax1.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="Gamma = 0.5 x Gamma_max (punto Kd)")
ax1.set_xlabel("[dCas9-Biotin] presentada (M)")
ax1.set_ylabel("Fraccion de ocupacion (Gamma_eq / Gamma_max)")
ax1.set_title("(a) Isoterma de equilibrio")
ax1.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())
ax1.set_ylim(-0.05, 1.05)
ax1.legend(loc="upper left", fontsize=8)
ax1.grid(alpha=0.3, which="both")

# --- (b) Cinetica de inmovilizacion: Gamma(t) en fmol/mm^2 vs tiempo en min (0-60 min) ---
Gamma_t_fmol = Gamma_t * 1e15   # mol/mm^2 -> fmol/mm^2
Gamma_max_fmol = Gamma_max * 1e15
ax2.plot(t_min_cinetico, Gamma_t_fmol, color="tab:green", linewidth=2, label="Gamma(t)")
ax2.axhline(0.90 * Gamma_max_fmol, color="tab:orange", linestyle="--", linewidth=1, label="90% Gamma_max")
ax2.axhline(0.95 * Gamma_max_fmol, color="black", linestyle="--", linewidth=1, label="95% Gamma_max")
ax2.set_xlabel("Tiempo de incubacion (min)")
ax2.set_ylabel("Gamma (fmol/mm^2)")
ax2.set_title("(b) Cinetica de inmovilizacion (0-60 min)")
ax2.legend(loc="lower right", fontsize=8)
ax2.grid(alpha=0.3)

# --- (c) ZOOM: primeros 30 segundos, donde ocurre la cinetica real ---
ventana_zoom_seg = 30.0
mascara_zoom = t_seg_cinetico <= ventana_zoom_seg
t_zoom_seg = t_seg_cinetico[mascara_zoom]
Gamma_zoom_fmol = Gamma_t_fmol[mascara_zoom]
ax3.plot(t_zoom_seg, Gamma_zoom_fmol, color="tab:green", linewidth=2, label="Gamma(t)")
ax3.axhline(0.90 * Gamma_max_fmol, color="tab:orange", linestyle="--", linewidth=1, label="90% Gamma_max")
ax3.axhline(0.95 * Gamma_max_fmol, color="black", linestyle="--", linewidth=1, label="95% Gamma_max")
ax3.set_xlabel("Tiempo de incubacion (s)")
ax3.set_ylabel("Gamma (fmol/mm^2)")
ax3.set_title(f"(c) ZOOM: primeros {ventana_zoom_seg:.0f} s")
ax3.legend(loc="lower right", fontsize=8)
ax3.grid(alpha=0.3)

# --- (d) Densidad de receptores en RU vs tiempo (0-60 min) ---
Gamma_t_RU = convertir_a_RU(Gamma_t, MW_dCas9)
Gamma_max_RU = convertir_a_RU(Gamma_max, MW_dCas9)
ax4.plot(t_min_cinetico, Gamma_t_RU, color="tab:purple", linewidth=2, label="Gamma(t) en RU")
ax4.axhline(Gamma_max_RU, color="gray", linestyle=":", linewidth=1, label="Gamma_max (RU)")
ax4.set_xlabel("Tiempo de incubacion (min)")
ax4.set_ylabel("Senal equivalente (RU)")
ax4.set_title("(d) Senal en unidades SPR (RU)")
ax4.legend(loc="lower right", fontsize=8)
ax4.grid(alpha=0.3)

# --- (e) Barrido de concentraciones: fraccion de ocupacion vs tiempo (log-log) ---
colores_barrido = ["tab:cyan", "tab:blue", "tab:green", "tab:olive", "tab:orange", "tab:red"]
for etiqueta, color in zip(etiquetas_concentracion, colores_barrido):
    ax5.plot(t_eval_barrido, resultados_barrido[etiqueta], color=color, linewidth=2, label=etiqueta)
ax5.set_xscale("log")
ax5.axhline(0.90, color="black", linestyle=":", linewidth=1, alpha=0.6)
ax5.set_xlabel("Tiempo de incubacion (s, escala log)")
ax5.set_ylabel("Fraccion de ocupacion (Gamma / Gamma_max)")
ax5.set_title("(e) Efecto de la concentracion en la velocidad")
ax5.set_ylim(-0.05, 1.05)
ax5.legend(loc="lower right", fontsize=7.5, title="[dCas9-Biotin]", ncol=2)
ax5.grid(alpha=0.3, which="both")

# --- Panel (f): sin uso, se deja desactivado para no sobrecargar la figura ---
ax6.axis("off")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("ModuloD_Langmuir_chip.png", dpi=150)
print("Figura guardada como: ModuloD_Langmuir_chip.png")
plt.show()


# =======================================================================
# OUTPUT PARA INTEGRACION CON MODULOS E Y F
# =======================================================================
print("=" * 70)
print("OUTPUT PARA INTEGRACION - MODULO D")
print("=" * 70)

Gamma_final_mol_mm2 = Gamma_t[-1]                         # [mol/mm^2]
Gamma_final_pmol_mm2 = Gamma_final_mol_mm2 * 1e12          # [pmol/mm^2]
Gamma_final_fmol_mm2 = Gamma_final_mol_mm2 * 1e15          # [fmol/mm^2]
Gamma_final_RU = convertir_a_RU(Gamma_final_mol_mm2, MW_dCas9)  # [RU]
fraccion_ocupacion_final = Gamma_final_mol_mm2 / Gamma_max

print(f"Gamma_final: {Gamma_final_mol_mm2:.4e} mol/mm^2  ({Gamma_final_pmol_mm2:.4f} pmol/mm^2 = {Gamma_final_fmol_mm2:.2f} fmol/mm^2)")
print(f"Gamma_final en unidades SPR: {Gamma_final_RU:.4f} RU")
print(f"Fraccion de ocupacion final (Gamma_final / Gamma_max): {fraccion_ocupacion_final*100:.2f}%")

t_90_seg = tiempo_para_alcanzar_fraccion(0.90, t_seg_cinetico, fraccion_ocupacion_t)
if t_90_seg is None:
    print(f"Tiempo para llegar a 90% de saturacion: NO alcanzado en {t_incubacion/60:.0f} min de incubacion")
else:
    print(f"Tiempo para llegar a 90% de saturacion: {t_90_seg/60:.2f} min ({t_90_seg:.1f} s)")

# Advertencia: la concentracion presentada no es suficiente para saturar la isoterma
if C_proteina < 10 * Kd:
    print("ADVERTENCIA: C_proteina < 10 x Kd -> la isoterma de equilibrio NO esta "
          "saturada a esta concentracion. Considerar aumentar C_proteina o revisar "
          "el valor que llega del Modulo B.")
else:
    print("C_proteina >= 10 x Kd: la isoterma de equilibrio predice saturacion "
          "practicamente completa a esta concentracion (esperado, dado el Kd "
          "extremadamente bajo de biotina-streptavidina).")

# Advertencia: la superficie quedo sub-saturada al final del tiempo de incubacion
if fraccion_ocupacion_final < 0.5:
    print("ADVERTENCIA: Gamma_final < 0.5 x Gamma_max -> la superficie del chip "
          "quedo SUB-SATURADA al final del tiempo de incubacion. Revisar "
          "C_proteina, kon, o extender el tiempo de incubacion.")
else:
    print("Gamma_final >= 0.5 x Gamma_max: la superficie alcanzo un nivel de "
          "saturacion razonable dentro del tiempo de incubacion modelado.")

# Valor que se pasa como Gamma_max_efectivo al Modulo E
print(f"Gamma_max_efectivo (input para Modulo E): {Gamma_final_mol_mm2:.4e} mol/mm^2")

print("=" * 70)


# =======================================================================
# NOTA: DIFERENCIA ENTRE ISOTERMA DE EQUILIBRIO Y MODELO CINETICO
# =======================================================================
# La ISOTERMA DE EQUILIBRIO (enfoque 1) responde a la pregunta:
#     "¿cuanto dCas9-Biotin se inmoviliza si dejo el sistema reaccionar
#      el tiempo suficiente, dada una concentracion [P] presentada?"
# Es la curva de saturacion clasica de Langmuir, y es la herramienta
# correcta para decidir QUE CONCENTRACION de proteina usar en el chip.
# No depende del tiempo: asume que ya se alcanzo el equilibrio.
#
# El MODELO CINETICO (enfoque 2) responde a la pregunta:
#     "¿cuanto TIEMPO de incubacion necesito para que la superficie del
#      chip se llene, dada una concentracion fija de proteina?"
# Es el enfoque relevante para el protocolo experimental real de SPR,
# donde se monitorea la senal en tiempo real durante la inyeccion de
# la muestra.
#
# En este proyecto, dado que Kd = 1e-15 M es extremadamente bajo
# comparado con las concentraciones de trabajo (~1e-7 M), la isoterma
# de equilibrio predice saturacion casi completa (~100%) practicamente
# sin importar la concentracion exacta que se use. Por eso el modelo
# CINETICO es el mas informativo en la practica: determina si el tiempo
# de incubacion estandar (1 hora) es suficiente para alcanzar ese
# equilibrio, o si el experimento esta limitado por la velocidad de
# union (kon) mas que por la afinidad (Kd).
# =======================================================================


# =======================================================================
# ESCENARIOS DE C_proteina -- INCERTIDUMBRE HEREDADA DE MODULO A v2.0 / B
# =======================================================================
# Al igual que en el Modulo B, se corre el modelo cinetico para los 3
# escenarios (pesimista/nominal/optimista) heredados del Monte Carlo del
# Modulo A v2.0, en vez de reportar un unico resultado como si fuera cierto.

def correr_cinetica_chip(C_proteina_i):
    """Resuelve la cinetica de Langmuir del chip para una concentracion dada."""
    sol_i = solve_ivp(
        fun=modelo_langmuir_cinetico, t_span=(0, t_incubacion), y0=[Gamma_0],
        method="RK45", t_eval=t_eval_cinetico, args=(kon, koff, C_proteina_i, Gamma_max),
        rtol=1e-8, atol=1e-22,
    )
    Gamma_i = np.clip(sol_i.y[0], 0.0, Gamma_max)
    fraccion_i = Gamma_i / Gamma_max
    t90_i = tiempo_para_alcanzar_fraccion(0.90, sol_i.t, fraccion_i)
    return {
        "t_seg": sol_i.t, "Gamma": Gamma_i, "fraccion": fraccion_i,
        "Gamma_final": Gamma_i[-1], "fraccion_final": fraccion_i[-1],
        "t90_seg": t90_i, "C_proteina": C_proteina_i,
    }


print("\n" + "=" * 70)
print("ESCENARIOS (incertidumbre heredada de Modulo A v2.0 / Modulo B)")
print("=" * 70)

escenarios_D = {
    "Pesimista (P5)": correr_cinetica_chip(C_proteina_pesimista),
    "Nominal": correr_cinetica_chip(C_proteina),
    "Optimista (P95)": correr_cinetica_chip(C_proteina_optimista),
}

print(f"{'Escenario':<18s}{'C_proteina (nM)':>16s}{'Fraccion final':>16s}{'Gamma_final (RU)':>18s}{'t90 (s)':>10s}")
print("-" * 78)
for nombre, r in escenarios_D.items():
    t90_txt = f"{r['t90_seg']:.2f}" if r["t90_seg"] is not None else "no alcanzado"
    Gamma_final_RU_i = convertir_a_RU(r["Gamma_final"], MW_dCas9)
    print(f"{nombre:<18s}{r['C_proteina']*1e9:>16.2f}{r['fraccion_final']*100:>15.1f}%{Gamma_final_RU_i:>18.1f}{t90_txt:>10s}")

print("\nInterpretacion: en los 3 escenarios la superficie del chip se satura")
print("practicamente por completo (fraccion final ~100%) dentro de la incubacion de")
print("1h, porque incluso en el escenario pesimista (5.78 nM) kon*C sigue siendo")
print(">>koff. La diferencia entre escenarios esta en la VELOCIDAD (t90 varia entre")
print("escenarios) mas que en el resultado final tras 1h completa -- pero si el")
print("equipo usa un tiempo de incubacion mas corto, la diferencia en fraccion")
print("alcanzada SI seria relevante entre escenarios.")

# --- Figura nueva: comparacion de escenarios ---
fig_esc_D, axes_esc_D = plt.subplots(1, 2, figsize=(13, 5.5))
fig_esc_D.suptitle("Modulo D - Escenarios de incertidumbre heredados de Modulo A v2.0 / B", fontsize=13, fontweight="bold")

colores_escenario_D = {"Pesimista (P5)": "tab:red", "Nominal": "tab:blue", "Optimista (P95)": "tab:green"}

ax_d1 = axes_esc_D[0]
for nombre, r in escenarios_D.items():
    ax_d1.plot(r["t_seg"], r["fraccion"] * 100, color=colores_escenario_D[nombre], linewidth=2, label=nombre)
ax_d1.set_xlabel("Tiempo de incubacion (s)")
ax_d1.set_ylabel("Fraccion de ocupacion (%)")
ax_d1.set_title("Saturacion del chip: los 3 escenarios (0-1h completa)")
ax_d1.legend(fontsize=9)
ax_d1.grid(alpha=0.3)

ax_d2 = axes_esc_D[1]
nombres_barra_D = list(escenarios_D.keys())
valores_barra_D = [convertir_a_RU(escenarios_D[n]["Gamma_final"], MW_dCas9) for n in nombres_barra_D]
colores_barra_D = [colores_escenario_D[n] for n in nombres_barra_D]
barras_D = ax_d2.bar(nombres_barra_D, valores_barra_D, color=colores_barra_D, alpha=0.8, edgecolor="black")
for barra, valor in zip(barras_D, valores_barra_D):
    ax_d2.text(barra.get_x() + barra.get_width() / 2, valor, f"{valor:.0f} RU", ha="center", va="bottom", fontsize=9)
ax_d2.axhline(Gamma_max_RU, color="gray", linestyle=":", linewidth=1, label=f"Gamma_max = {Gamma_max_RU:.0f} RU")
ax_d2.set_ylabel("Senal final (RU)")
ax_d2.set_title("Senal final tras 1h de incubacion, por escenario")
ax_d2.legend(fontsize=8)
ax_d2.grid(alpha=0.3, axis="y")

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("ModuloD_escenarios_ModuloA_v2.png", dpi=150)
print("\nFigura guardada como: ModuloD_escenarios_ModuloA_v2.png")
plt.show()


# =======================================================================
# SECCION DE INTEGRACION CON MODULOS E Y F
# =======================================================================
#
# En el modelo integrado completo (Modulo A -> B -> C -> D -> E -> F):
#
#   1. C_proteina (seccion de parametros de este script) ACTUALMENTE viene
#      DIRECTAMENTE del Modulo B (que a su vez hereda del Modulo A v2.0),
#      como proxy temporal mientras el Modulo C (purificacion) no este
#      implementado. Cuando exista el Modulo C real, C_proteina debe
#      reemplazarse por P_funcional (su output), que probablemente sera
#      MENOR a los valores usados aqui una vez se incluya el rendimiento
#      de purificacion (yield_purif, estimado ~0.55 en el modelo de
#      Valeria, todavia sin implementar formalmente en este pipeline).
#
#   2. Gamma_final (calculado en la seccion de OUTPUT PARA INTEGRACION) se
#      pasa como Gamma_max_efectivo al Modulo E, usando el escenario
#      NOMINAL. El Modulo E deberia, idealmente, tambien correr sus 3
#      escenarios correspondientes (ver ModuloD_escenarios_ModuloA_v2.png)
#      en vez de un unico valor, siguiendo el mismo patron que B y D.
#
#   3. Gamma_final tambien determina la SENAL MAXIMA ESPERADA en el
#      Modulo F (respuesta del biosensor), ya que la magnitud de la
#      senal de deteccion es proporcional a la cantidad de receptores
#      funcionales inmovilizados en el chip.
#
#   4. Si el equipo de WetLab obtiene el valor real de Gamma_max
#      (especificaciones del fabricante del chip, o experimento propio
#      de saturacion en SPR), el UNICO parametro que debe cambiarse en
#      este script es Gamma_max (seccion de parametros). Todo el resto
#      del modelo (isoterma, cinetica, conversion a RU) se recalcula
#      automaticamente con el nuevo valor.
#
#   5. Si el equipo mide kon y koff especificos de su chip/lote (en vez de
#      los valores de literatura corregidos en esta version), actualizar
#      kon y koff directamente; Kd se recalcula automaticamente como
#      koff/kon para mantener la consistencia interna del modelo.
#
# =======================================================================