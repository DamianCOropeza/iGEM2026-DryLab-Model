#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=======================================================================
MODULO B - CINETICA DE BIOTINILACION DE dCas9-AviTag POR BirA
=======================================================================
Proyecto iGEM - Modelado matematico de biologia sintetica

Este script simula la biotinilacion enzimatica de la proteina de fusion
dCas9-AviTag por la enzima BirA (biotina ligasa) en E. coli, inducida
a 16 grados C durante 24 horas.

# ============================================================
# INSTALACION DE DEPENDENCIAS (ejecutar en terminal una sola vez)
# ============================================================
# Si usas conda (recomendado para este proyecto):
#   conda activate iGEM
#   conda install numpy scipy matplotlib
#
# Verificar instalacion:
#   python -c "import numpy, scipy, matplotlib; print('OK')"
#
# Ejecutar el script:
#   python modulo_B_biotinilacion.py
# ============================================================

=======================================================================
SUPUESTOS DEL MODELO
=======================================================================
1. ATP no es limitante: [ATP]_intracelular >> Km_ATP (~0.25 mM). Por lo
   tanto el ATP no se incluye como variable dinamica en la ecuacion de
   velocidad; se asume que la reaccion de activacion de biotina
   (Paso 1: BirA + Biotina + ATP -> BirA-Bio-AMP + PPi) nunca esta
   limitada por falta de ATP.
2. BirA es catalitica: no se consume en la reaccion neta, por lo que su
   concentracion se mantiene CONSTANTE durante toda la simulacion.
3. La correccion de temperatura mediante el factor Q10 es una
   EXTRAPOLACION matematica a partir de datos cineticos medidos a 37 C
   (o temperatura ambiente), NO es una medicion directa a 16 C. Debe
   validarse experimentalmente cuando sea posible.
4. Modelo secuencial: se asume que al iniciar el Modulo B, tanto BirA
   como dCas9-AviTag ya fueron producidos por la maquinaria de
   traduccion (esto se modela en el Modulo A). Las concentraciones
   iniciales de este modulo (BirA_0, S_0) representan el estado del
   sistema en t = 20 h del Modulo A.
5. La biotina intracelular libre se inicializa en 5 uM, dentro del
   rango reportado en literatura (1-10 uM). Este valor es sensible y
   se recomienda un analisis de sensibilidad posterior.
=======================================================================
"""

# =======================================================================
# 1. IMPORTS Y CONFIGURACION
# =======================================================================
import numpy as np                       # manejo de arreglos numericos y operaciones vectoriales
from scipy.integrate import solve_ivp    # resolvedor de sistemas de ecuaciones diferenciales ordinarias (ODEs)
import matplotlib.pyplot as plt          # generacion de graficas para visualizar los resultados

# =======================================================================
# 2. PARAMETROS DEL MODELO
# =======================================================================

# --- Parametros cineticos de BirA (temperatura de referencia: 37 C) ---
kcat_ref = 5.4        # s^-1 | Fuente: Cronan lab, Biochemistry; confirmado por
                       #        Avidity Inc. (AviTag biotinylated 2x faster than BCCP)
                       #        Rango literatura: 5-11 s^-1

Km_S = 2e-6            # M   | Km de BirA para AviTag peptide (~1-5 uM)
                       #        Fuente: Schatz 1993; Biotin ligase characterization
                       #        (ScienceDirect doi:10.1016/j.ab.2008.01.019)

Km_bio = 1.45e-6        # M   | Km de BirA para biotina libre (~1.45 uM)
                       #        Fuente: Cronan & Reed, Methods Enzymol. 2000

# --- Biotina intracelular en E. coli ---
bio_intra = 5e-6       # M   | Concentracion intracelular de biotina libre en E. coli
                       #        Rango literatura: 1-10 uM. Valor medio: 5 uM.
                       #        Fuente: Cronan 1989, J Biol Chem; Lin et al. 2010
                       #        NOTA: puede ser limitante si [BirA] es alta

# --- Corriccion de temperatura (Q10) ---
T_ref = 37.0           # C    | Temperatura de referencia de los parametros reportados
T_exp = 16.0           # C    | Temperatura de induccion experimental
Q10 = 2.0              # adim | Factor Q10 estandar para enzimas (rango: 1.5-3.0)
                       #        Supuesto documentado: mismo Q10 que en Modulo A
                       #        Fuente: referencia general de cinetica enzimatica
                       #        Atkinson 1977; aplicado como extrapolacion

# --- Condiciones iniciales del Modulo A (valores de ejemplo) ---
# En el modelo integrado, estos valores vienen del output del Modulo A a t=20h.
# Para esta version standalone, usar valores representativos:
BirA_0 = 1.0e-6         # M   | [BirA] inicial ~ output Modulo A a t=20h  # <- Modulo A
                       #        BirA es catalitica: su concentracion NO cambia
S_0 = 8.0e-6            # M   | [dCas9-AviTag] inicial ~ output Modulo A a t=20h  # <- Modulo A
                       #        (fraccion de dCas9 total que no esta aun biotinilada)

# --- Tiempo de simulacion ---
t_start = 0            # s
t_end = 86400           # s   | 24 horas (puede ajustarse; la biotinilacion in vivo
                       #        ocurre concurrentemente con la expresion)


# =======================================================================
# 3. CORRECCION DE TEMPERATURA Q10
# =======================================================================
def ajustar_temperatura(k_ref, Q10, T_exp, T_ref):
    """
    Ajusta una constante cinetica medida a T_ref hacia una nueva
    temperatura T_exp utilizando el factor Q10.

    Formula: k_ajustado = k_ref * Q10 ^ ((T_exp - T_ref) / 10)

    Parametros:
        k_ref  : constante cinetica a la temperatura de referencia
        Q10    : factor de sensibilidad termica de la enzima (adimensional)
        T_exp  : temperatura experimental (C)
        T_ref  : temperatura de referencia de la literatura (C)

    Retorna:
        k_ajustado : constante cinetica corregida a T_exp
    """
    k_ajustado = k_ref * (Q10 ** ((T_exp - T_ref) / 10.0))
    return k_ajustado


# Se aplica la correccion Q10 al kcat de BirA para pasar de 37 C a 16 C
kcat_adj = ajustar_temperatura(kcat_ref, Q10, T_exp, T_ref)

print("=" * 60)
print("MODULO B - Biotinilacion de dCas9-AviTag por BirA")
print("=" * 60)
print(f"kcat a 37 C: {kcat_ref:.2f} s-1 -> kcat ajustado a 16 C: {kcat_adj:.4f} s-1")


# =======================================================================
# 4. SISTEMA DE ODEs
# =======================================================================
def modelo_biotinilacion(t, y, params):
    """
    Define el sistema de ecuaciones diferenciales que describe la
    biotinilacion enzimatica de dCas9-AviTag por BirA.

    Variables de estado (vector y):
        y[0] = S        -> [dCas9-AviTag] sin biotinilar (M)
        y[1] = P        -> [dCas9-Biotin] producto biotinilado (M)
        y[2] = biotina  -> [biotina libre intracelular] (M)

    La velocidad de reaccion v sigue una cinetica Michaelis-Menten de
    dos sustratos (dCas9-AviTag y biotina), con BirA actuando como
    catalizador de concentracion constante:

        v = kcat_adj * [BirA] * (S / (Km_S + S)) * (biotina / (Km_bio + biotina))
    """
    # Se protegen las concentraciones contra valores negativos que
    # pueden aparecer por error numerico del integrador
    S = max(y[0], 0.0)
    P = max(y[1], 0.0)
    biotina = max(y[2], 0.0)

    BirA_conc = params["BirA_0"]      # BirA es catalitica: concentracion constante
    kcat_local = params["kcat_adj"]
    Km_S_local = params["Km_S"]
    Km_bio_local = params["Km_bio"]

    # Velocidad de biotinilacion (M/s), doble Michaelis-Menten
    v = kcat_local * BirA_conc * (S / (Km_S_local + S)) * (biotina / (Km_bio_local + biotina))

    # dS/dt: el sustrato dCas9-AviTag se consume a medida que se biotinila
    dS_dt = -v

    # dP/dt: el producto dCas9-Biotin se acumula a la misma velocidad v
    dP_dt = v

    # d(biotina)/dt: cada evento de biotinilacion consume una molecula de biotina
    dbiotina_dt = -v

    return [dS_dt, dP_dt, dbiotina_dt]


# =======================================================================
# 5. RESOLUCION NUMERICA
# =======================================================================

# Diccionario de parametros que se pasa a la funcion del modelo
parametros = {
    "BirA_0": BirA_0,
    "kcat_adj": kcat_adj,
    "Km_S": Km_S,
    "Km_bio": Km_bio,
}

# Condiciones iniciales: [S_0, P_0=0, biotina_0]
y0 = [S_0, 0.0, bio_intra]

# Puntos de tiempo donde queremos evaluar la solucion (para graficas suaves)
t_eval = np.linspace(t_start, t_end, 2000)

sol = solve_ivp(
    fun=modelo_biotinilacion,
    t_span=(t_start, t_end),
    y0=y0,
    method="RK45",
    t_eval=t_eval,
    dense_output=True,
    args=(parametros,),
    # NOTA TECNICA: las concentraciones de este sistema son del orden de
    # 1e-6 M. La tolerancia absoluta por defecto de solve_ivp (atol=1e-6)
    # es del mismo orden que las propias variables de estado, lo cual
    # puede causar que el integrador "se pase" (overshoot) hacia valores
    # negativos cuando una especie se agota (ej. biotina llegando a 0).
    # Por eso se usan tolerancias mas estrictas, apropiadas a la escala
    # micromolar del sistema:
    rtol=1e-8,
    atol=1e-12,
)

# Verificacion de convergencia del integrador
if sol.success:
    print("Integracion numerica: EXITOSA (RK45 convergio correctamente)")
else:
    print("ADVERTENCIA: la integracion numerica NO convergio.")
    print(f"Mensaje del solver: {sol.message}")

# Extraccion de resultados
t_seg = sol.t                 # tiempo en segundos
S_t = np.maximum(sol.y[0], 0.0)         # [dCas9-AviTag] sin biotinilar (clip de seguridad >= 0)
P_t = np.maximum(sol.y[1], 0.0)         # [dCas9-Biotin] (clip de seguridad >= 0)
biotina_t = np.maximum(sol.y[2], 0.0)   # [biotina libre] (clip de seguridad >= 0)

t_horas = t_seg / 3600.0        # conversion de segundos a horas para graficar


# =======================================================================
# 6. CALCULO DE % BIOTINILACION
# =======================================================================

# Porcentaje de biotinilacion respecto al total inicial de dCas9-AviTag
pct_bio = (P_t / S_0) * 100.0

pct_bio_final = pct_bio[-1]
print(f"% biotinilacion a 24h: {pct_bio_final:.1f}%")

# Puntos de control en horas especificas
puntos_control_h = [4, 8, 12, 20]
for hora in puntos_control_h:
    # Se busca el indice de tiempo mas cercano a la hora deseada
    idx_cercano = np.argmin(np.abs(t_horas - hora))
    print(f"% biotinilacion a {hora}h: {pct_bio[idx_cercano]:.1f}%")

print("=" * 60)


# --- Marcadores de tiempo para las graficas (1h, 2h, 4h, 8h, 12h, 20h, 24h) ---
# Se usa la solucion densa (sol.sol) para evaluar el estado exacto del
# sistema en cada uno de estos tiempos, en vez de tomar el punto mas
# cercano de la malla t_eval.
marker_horas = np.array([1, 2, 4, 8, 12, 20, 24])
marker_segundos = marker_horas * 3600.0
marker_segundos = np.clip(marker_segundos, t_start, t_end)  # por si t_end fuera menor a 24h
estado_marcadores = sol.sol(marker_segundos)   # shape (3, len(marker_horas))
S_marker = np.maximum(estado_marcadores[0], 0.0)
P_marker = np.maximum(estado_marcadores[1], 0.0)
biotina_marker = np.maximum(estado_marcadores[2], 0.0)
pct_bio_marker = (P_marker / S_0) * 100.0


def tiempo_para_alcanzar_pct(pct_objetivo, t_horas_arr, pct_bio_arr):
    """
    Interpola el tiempo (en horas) en el que la curva de % biotinilacion
    cruza un valor objetivo (ej. 50%, 80%, 95%).

    Retorna None si el objetivo nunca se alcanza dentro de la simulacion.
    """
    if pct_bio_arr[-1] < pct_objetivo:
        return None
    # np.interp requiere que la variable independiente (pct_bio_arr) sea
    # creciente; esto se cumple porque P(t) es monotona creciente.
    t_cruce = np.interp(pct_objetivo, pct_bio_arr, t_horas_arr)
    return t_cruce


def anadir_margen_ylim(ax, datos, margen_frac=0.05):
    """
    Ajusta el limite del eje Y del eje 'ax' para dejar un margen
    proporcional (margen_frac) arriba y abajo del rango real de 'datos'.
    """
    y_min = np.min(datos)
    y_max = np.max(datos)
    rango = y_max - y_min
    if rango == 0:
        # Si los datos son constantes, se usa un margen absoluto pequeno
        # para que la linea no quede pegada al borde del grafico
        rango = abs(y_max) if y_max != 0 else 1.0
    margen = rango * margen_frac
    ax.set_ylim(y_min - margen, y_max + margen)


# =======================================================================
# 7. VISUALIZACION
# =======================================================================
# Se usa una grilla 2x2: (a) sustrato/producto en 24h, (b) biotina libre,
# (c) % biotinilacion con lineas de referencia, y (d) zoom de las
# primeras 4h donde ocurre la mayor parte de la reaccion.

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle("Modulo B - Cinetica de Biotinilacion por BirA (16 C, 24 h)", fontsize=14, fontweight="bold")
ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

# --- Grafica (a): dCas9-AviTag y dCas9-Biotin vs tiempo (0-24h) ---
ax1.plot(t_horas, S_t * 1e6, label="dCas9-AviTag (sin biotinilar)", color="tab:orange", linewidth=2)
ax1.plot(t_horas, P_t * 1e6, label="dCas9-Biotin (biotinilado)", color="tab:green", linewidth=2)
ax1.plot(marker_horas, S_marker * 1e6, "o", color="tab:orange", markersize=6, markeredgecolor="black", markeredgewidth=0.5, zorder=5)
ax1.plot(marker_horas, P_marker * 1e6, "o", color="tab:green", markersize=6, markeredgecolor="black", markeredgewidth=0.5, zorder=5)
ax1.set_xlabel("Tiempo (h)")
ax1.set_ylabel("Concentracion (uM)")
ax1.set_title("Sustrato y producto (0-24h)")
ax1.legend(loc="best", fontsize=9)
ax1.grid(alpha=0.3)
anadir_margen_ylim(ax1, np.concatenate([S_t, P_t]) * 1e6)

# --- Grafica (b): Biotina libre vs tiempo ---
ax2.plot(t_horas, biotina_t * 1e6, color="tab:blue", linewidth=2)
ax2.plot(marker_horas, biotina_marker * 1e6, "o", color="tab:blue", markersize=6, markeredgecolor="black", markeredgewidth=0.5, zorder=5)
ax2.set_xlabel("Tiempo (h)")
ax2.set_ylabel("Biotina libre (uM)")
ax2.set_title("Biotina intracelular disponible")
ax2.grid(alpha=0.3)
anadir_margen_ylim(ax2, biotina_t * 1e6)

# --- Grafica (c): % Biotinilacion vs tiempo, con lineas de referencia ---
ax3.plot(t_horas, pct_bio, color="tab:purple", linewidth=2, label="% biotinilacion")
ax3.plot(marker_horas, pct_bio_marker, "o", color="tab:purple", markersize=6, markeredgecolor="black", markeredgewidth=0.5, zorder=5)
ax3.axhline(50, color="tab:gray", linestyle="--", linewidth=1)
ax3.axhline(80, color="tab:red", linestyle="--", linewidth=1)
ax3.axhline(95, color="black", linestyle="--", linewidth=1)
ax3.text(t_horas[-1], 50, " Target 50%", va="center", ha="right", fontsize=8, color="tab:gray")
ax3.text(t_horas[-1], 80, " Target 80%", va="bottom", ha="right", fontsize=8, color="tab:red")
ax3.text(t_horas[-1], 95, " Target 95%", va="bottom", ha="right", fontsize=8, color="black")
ax3.set_xlabel("Tiempo (h)")
ax3.set_ylabel("% Biotinilacion")
ax3.set_title("Progreso de biotinilacion")
ax3.legend(loc="lower right", fontsize=9)
ax3.grid(alpha=0.3)
anadir_margen_ylim(ax3, np.concatenate([pct_bio, [0, 100]]))

# --- Grafica (d): ZOOM a las primeras 4 horas (sustrato y producto) ---
ventana_zoom_h = 4.0
mascara_zoom = t_horas <= ventana_zoom_h
t_zoom = t_horas[mascara_zoom]
S_zoom = S_t[mascara_zoom] * 1e6
P_zoom = P_t[mascara_zoom] * 1e6

ax4.plot(t_zoom, S_zoom, label="dCas9-AviTag (sin biotinilar)", color="tab:orange", linewidth=2)
ax4.plot(t_zoom, P_zoom, label="dCas9-Biotin (biotinilado)", color="tab:green", linewidth=2)

# Solo se muestran los marcadores que caen dentro de la ventana de zoom (1h, 2h, 4h)
mascara_marker_zoom = marker_horas <= ventana_zoom_h
ax4.plot(marker_horas[mascara_marker_zoom], S_marker[mascara_marker_zoom] * 1e6, "o", color="tab:orange", markersize=6, markeredgecolor="black", markeredgewidth=0.5, zorder=5)
ax4.plot(marker_horas[mascara_marker_zoom], P_marker[mascara_marker_zoom] * 1e6, "o", color="tab:green", markersize=6, markeredgecolor="black", markeredgewidth=0.5, zorder=5)

ax4.set_xlabel("Tiempo (h)")
ax4.set_ylabel("Concentracion (uM)")
ax4.set_title(f"ZOOM: primeras {ventana_zoom_h:.0f}h (fase activa de la reaccion)")
ax4.legend(loc="best", fontsize=9)
ax4.grid(alpha=0.3)
anadir_margen_ylim(ax4, np.concatenate([S_zoom, P_zoom]))

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("ModuloB_biotinilacion.png", dpi=150)
print("Figura guardada como: ModuloB_biotinilacion.png")
plt.show()


# =======================================================================
# 8. OUTPUT PARA INTEGRACION
# =======================================================================
# Resumen final de resultados, pensado para copiarse/leerse facilmente
# al conectar este modulo con el resto del pipeline (Modulo A -> B -> C).

print("=" * 60)
print("OUTPUT PARA INTEGRACION - MODULO B")
print("=" * 60)

# --- (1) % de biotinilacion a t=20h y t=24h ---
pct_20h = pct_bio_marker[marker_horas == 20][0]
pct_24h = pct_bio[-1]   # t_end de la simulacion es 24h
print(f"(1) % biotinilacion a t=20h: {pct_20h:.1f}%")
print(f"    % biotinilacion a t=24h: {pct_24h:.1f}%")

# --- (2) Tiempo en horas para alcanzar 50%, 80% y 95% de biotinilacion ---
for objetivo in [50, 80, 95]:
    t_cruce = tiempo_para_alcanzar_pct(objetivo, t_horas, pct_bio)
    if t_cruce is None:
        print(f"(2) Tiempo para alcanzar {objetivo}%: NO alcanzado en {t_end/3600:.0f}h de simulacion")
    else:
        print(f"(2) Tiempo para alcanzar {objetivo}%: {t_cruce:.3f} h ({t_cruce*3600:.1f} s)")

# --- (3) Concentracion final de dCas9-Biotin disponible para el Modulo C ---
P_final_uM = P_t[-1] * 1e6
print(f"(3) [dCas9-Biotin] final (input para Modulo C): {P_final_uM:.3f} uM")

# --- (4) Concentracion de biotina libre restante al final ---
biotina_final_uM = biotina_t[-1] * 1e6
print(f"(4) [Biotina libre] restante al final: {biotina_final_uM:.4f} uM")

# --- (5) Advertencia si la biotina se agoto antes del final de la simulacion ---
# Se considera "agotada" cuando cae por debajo del 1% de su valor inicial
umbral_agotamiento = 0.01 * bio_intra
indices_agotada = np.where(biotina_t <= umbral_agotamiento)[0]
if len(indices_agotada) > 0:
    t_agotamiento_h = t_horas[indices_agotada[0]]
    print(f"(5) ADVERTENCIA: la biotina libre se agoto (<1% del valor inicial) "
          f"a partir de t = {t_agotamiento_h:.3f} h. Esto puede estar limitando "
          f"la velocidad de biotinilacion; considerar aumentar bio_intra o "
          f"revisar el suministro de biotina en el medio de cultivo.")
else:
    print("(5) La biotina libre NO se agoto durante la simulacion (siempre > 1% del valor inicial).")

# --- (6) Valor de kcat ajustado a 16 C usado en la simulacion ---
print(f"(6) kcat ajustado a 16 C (usado en la simulacion): {kcat_adj:.4f} s-1")

print("=" * 60)


# =======================================================================
# 9. INTEGRACION CON MODULO A
# =======================================================================
#
# En el modelo integrado completo (Modulo A + Modulo B + Modulo C), este
# script NO debe usar los valores de ejemplo BirA_0 y S_0 definidos
# arriba. En su lugar:
#
#   1. El Modulo A resuelve su propio sistema de ODEs (expresion genica
#      de BirA y dCas9-AviTag) con solve_ivp, obteniendo las series de
#      tiempo P_BirA(t) y P_dCas9(t).
#   2. Se extraen los valores en t = 20 h:
#         BirA_0 = P_BirA_moduloA(t=20h)
#         S_0    = P_dCas9_moduloA(t=20h)
#   3. Estos valores se pasan como condiciones iniciales al presente
#      Modulo B (reemplazando las lineas BirA_0 = ... y S_0 = ... de la
#      seccion 2 de este script).
#   4. El Modulo B corre su propia integracion (0 a 24 h de induccion a
#      16 C) y produce pct_bio_final, el porcentaje final de dCas9
#      biotinilado.
#   5. pct_bio_final se pasa como input al Modulo C, que modelara la
#      union del sistema dCas9-Biotin-estreptavidina (o el paso
#      funcional siguiente del sistema, segun el diseño del proyecto).
#
# Output de este modulo -> Modulo C: pct_bio_final
# =======================================================================# =======================================================================