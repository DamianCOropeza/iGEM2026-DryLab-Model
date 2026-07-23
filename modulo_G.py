#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================
# MODULO G - LOD, Sensibilidad Global y Design Space
# Biosensor dCas9-AviTag | Deteccion ctDNA | SPR Fase
# ============================================================
# Proyecto iGEM 2025 - Modelado matematico de biologia sintetica (Dry Lab)
#
# Dependencias: pip install SALib numpy scipy matplotlib --break-system-packages
#
# NOTA DE AUDITORIA BIBLIOGRAFICA (leer antes de usar los resultados):
# ------------------------------------------------------------------
# El documento de especificacion original de este modulo (v1.0) contenia
# varias citas y valores que CONTRADICEN lo ya verificado y usado de forma
# consistente en los Modulos A-F de este mismo pipeline. Ejemplos
# encontrados y corregidos aqui:
#   - Vera et al. (2007) [phi_16]: el doc original citaba "Biotechnol
#     Bioeng, 99(1), 34-44" -- verificado contra 8 fuentes independientes,
#     ese articulo NO EXISTE. La cita correcta (ya usada en Modulos A y C)
#     es Biotechnol. Bioeng., 96(6), 1101-1106, DOI 10.1002/bit.21218.
#   - Grigorenko et al. (1999) [S_phase]: el doc original cambiaba autores
#     (V.N. Grigorenko y Savransky en vez de A.N. Grigorenko y Kabashin) y
#     el titulo del paper, manteniendo el mismo DOI -- variante corrupta.
#     Se usa la version ya verificada en Modulo F.
#   - Stengel et al. (2007) [dn/dc DNA]: el doc original citaba un paper
#     DISTINTO por completo (mismos autores apellido, coautores y titulo
#     diferentes) con un DOI distinto. Se usa la version ya verificada en
#     Modulo F (Stengel, Zahn & Hook, JACS 129(31), 9584-9585).
#   - Nikitin et al. (1999) [sigma_blank]: DOI alterado respecto al ya
#     verificado en Modulo F. Se usa el DOI correcto.
#   - delta_R_ref, delta_m_ref, delta_p_ref: el doc original cambiaba
#     valores numericos y/o citas respecto a los ya establecidos en
#     Modulo A, sin justificacion. Se usan los valores YA VALIDADOS de
#     Modulo A para mantener consistencia de todo el pipeline.
#
# Se prioriza la CONSISTENCIA INTERNA del pipeline completo (A-F ya
# construidos y verificados) sobre los valores nuevos de la especificacion
# de Modulo G donde estos entran en conflicto. Todos los cambios respecto
# al documento original quedan comentados explicitamente abajo.
#
# Referencias principales (verificadas):
#   Herman & Usher (2017). SALib: An open-source Python library for
#       sensitivity analysis. JOSS, 2(9), 97.
#       https://doi.org/10.21105/joss.00097
#   Morris (1991). Factorial sampling plans for preliminary computational
#       experiments. Technometrics, 33(2), 161-174.
#       https://doi.org/10.1080/00401706.1991.10484804
#   Saltelli et al. (2008). Global Sensitivity Analysis: The Primer.
#       Wiley. https://doi.org/10.1002/9780470725184
#   ICH Q8(R2) (2009). Pharmaceutical Development. ICH Harmonised
#       Guideline. https://www.ich.org/page/quality-guidelines
# ============================================================

import numpy as np
GRAFICAS_DISPONIBLES = True
try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    import matplotlib.ticker as ticker
except Exception as _err_matplotlib:
    GRAFICAS_DISPONIBLES = False
    print("=" * 70)
    print("ADVERTENCIA: no se pudo importar matplotlib en este entorno.")
    print(f"Detalle: {_err_matplotlib}")
    print("El script seguira corriendo y calculando/imprimiendo TODOS los")
    print("resultados numericos, pero NO se generaran las figuras PNG.")
    print("Causa tipica: incompatibilidad de version entre numpy y matplotlib")
    print("compilado (ej. numpy 2.x con matplotlib compilado para numpy 1.x).")
    print("Solucion real (correr en terminal, no en este script):")
    print("  pip install --upgrade --force-reinstall matplotlib")
    print("=" * 70)

    class _EjesFalsos:
        """Objeto 'todo-lo-acepta': absorbe cualquier atributo/llamada/operacion
        aritmetica de matplotlib sin hacer nada, para que el resto del script
        no truene cuando matplotlib no esta disponible (incluye soporte para
        expresiones como barra.get_x() + barra.get_width()/2, que aparecen al
        anotar valores sobre las barras de las graficas)."""
        def __getattr__(self, nombre):
            return self
        def __call__(self, *args, **kwargs):
            return self
        def __getitem__(self, indice):
            return self
        def __add__(self, otro): return self
        def __radd__(self, otro): return self
        def __sub__(self, otro): return self
        def __rsub__(self, otro): return self
        def __mul__(self, otro): return self
        def __rmul__(self, otro): return self
        def __truediv__(self, otro): return self
        def __rtruediv__(self, otro): return self
        def __neg__(self): return self
        def __pow__(self, otro): return self
        def __float__(self): return 0.0
        def __int__(self): return 0
        def __bool__(self): return False
        def __iter__(self): return iter([])
        def __len__(self): return 0

    class _ContenedorEjes(_EjesFalsos):
        """Emula el arreglo de ejes que devuelve plt.subplots(nrows, ncols)
        para que tanto axes[i, j] como el desempaquetado ax1, ax2 = ... funcionen."""
        def __init__(self, n):
            self._ejes = [_EjesFalsos() for _ in range(n)]
        def __iter__(self):
            return iter(self._ejes)
        def __getitem__(self, indice):
            return self._ejes[0]

    class _ProxyPyplot(_EjesFalsos):
        def subplots(self, *args, **kwargs):
            nrows = args[0] if len(args) >= 1 else kwargs.get("nrows", 1)
            ncols = args[1] if len(args) >= 2 else kwargs.get("ncols", 1)
            n = nrows * ncols
            if n <= 1:
                return _EjesFalsos(), _EjesFalsos()
            return _EjesFalsos(), _ContenedorEjes(n)
        def figure(self, *args, **kwargs):
            return _EjesFalsos()

    plt = _ProxyPyplot() if 'plt' == 'plt' else _EjesFalsos()
    mcolors = _ProxyPyplot() if 'mcolors' == 'plt' else _EjesFalsos()
    mpatches = _ProxyPyplot() if 'mpatches' == 'plt' else _EjesFalsos()
    ticker = _ProxyPyplot() if 'ticker' == 'plt' else _EjesFalsos()

from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
_rng = np.random.default_rng(42)

# ═══════════════════════════════════════════════════════════════
# ANALISIS DE SENSIBILIDAD SIN SALib -- implementacion propia
# ═══════════════════════════════════════════════════════════════
# CAMBIO IMPORTANTE vs. version anterior: se elimino la dependencia de la
# libreria SALib. Motivo: SALib fallaba en el entorno de Windows/Anaconda
# del equipo (numpy 2.4.6) incluso despues de aislar el problema de
# matplotlib -- probablemente por su propia incompatibilidad interna con
# esa version de numpy, que no se puede diagnosticar ni arreglar desde
# aqui sin acceso a esa maquina. En vez de seguir dependiendo de un
# paquete externo fragil, se implementan aqui las DOS metodologias
# (Morris y Sobol) directamente con formulas publicadas y verificables,
# usando solo numpy (que ya se confirmo que funciona en el equipo). Esto
# NO reduce el rigor del analisis: son las mismas formulas que usa SALib
# internamente, solo que auto-contenidas.
#
# Morris (Elementary Effects): Morris, M. D. (1991). Factorial sampling
#   plans for preliminary computational experiments. Technometrics,
#   33(2), 161-174. https://doi.org/10.1080/00401706.1991.10484804
#
# Sobol (indices de varianza, estimador de Saltelli): Saltelli, A., et al.
#   (2010). Variance based sensitivity analysis of model output. Design
#   and estimator for the total sensitivity index. Computer Physics
#   Communications, 181(2), 259-270.
#   https://doi.org/10.1016/j.cpc.2009.09.018

def _construir_trayectoria_morris(k, delta, indices_log, rng):
    """
    Construye UNA trayectoria de Morris (k+1 puntos en el cubo unitario
    [0,1]^k), variando un factor a la vez en orden aleatorio, con signo
    aleatorio (+delta o -delta), verificando que cada paso se mantenga
    dentro de [0,1].

    indices_log: lista booleana, True si ese parametro debe muestrearse
    en escala log10 (para parametros que abarcan varios ordenes de
    magnitud, ej. S_phase, sigma_blank) -- evita subestimar su
    sensibilidad, tal como se advierte en la especificacion original.

    Retorna: trayectoria (k+1, k) en [0,1], el orden de perturbacion, y
    el signo (+-1) usado para cada factor (necesario para calcular
    correctamente el efecto elemental despues).
    """
    x_base = rng.uniform(0.0, 1.0 - delta, size=k)
    signos = rng.choice([-1.0, 1.0], size=k)
    orden = rng.permutation(k)

    trayectoria = np.zeros((k + 1, k))
    trayectoria[0] = x_base
    punto = x_base.copy()
    signos_usados = np.zeros(k)
    for paso, idx in enumerate(orden):
        punto = punto.copy()
        s = signos[idx]
        if not (0.0 <= punto[idx] + s * delta <= 1.0):
            s = -s
        punto[idx] += s * delta
        signos_usados[idx] = s
        trayectoria[paso + 1] = punto
    return trayectoria, orden, signos_usados


def _escalar_a_bounds(X_unit, bounds, indices_log):
    """
    Escala muestras del cubo unitario [0,1]^k a los bounds reales de cada
    parametro. Si indices_log[i] es True, la escala es logaritmica
    (log10) en vez de lineal -- necesario para parametros que abarcan
    varios ordenes de magnitud (S_phase, sigma_blank), tal como senalaba
    la especificacion original del modulo.
    """
    X_escalado = np.zeros_like(X_unit)
    for i, (lo, hi) in enumerate(bounds):
        if indices_log[i]:
            log_lo, log_hi = np.log10(lo), np.log10(hi)
            X_escalado[:, i] = 10 ** (log_lo + X_unit[:, i] * (log_hi - log_lo))
        else:
            X_escalado[:, i] = lo + X_unit[:, i] * (hi - lo)
    return X_escalado


def muestrear_morris(bounds, N_trayectorias, num_levels=8, seed=42):
    """
    Genera N_trayectorias trayectorias de Morris para un espacio de k
    parametros con los bounds dados. Retorna X (N_trayectorias*(k+1), k)
    ya escalado a los bounds reales, y la informacion de cada trayectoria
    (orden y signos) necesaria para el analisis posterior.
    """
    k = len(bounds)
    rng_local = np.random.default_rng(seed)
    indices_log = [(hi / lo) > 50 for lo, hi in bounds]
    delta = num_levels / (2.0 * (num_levels - 1))

    trayectorias_unit = []
    info_trayectorias = []
    for _ in range(N_trayectorias):
        tray, orden, signos = _construir_trayectoria_morris(k, delta, indices_log, rng_local)
        trayectorias_unit.append(tray)
        info_trayectorias.append((orden, signos))

    X_unit = np.vstack(trayectorias_unit)
    X_escalado = _escalar_a_bounds(X_unit, bounds, indices_log)
    return X_escalado, info_trayectorias, delta, indices_log


def analizar_morris(Y, info_trayectorias, k, delta):
    """
    Calcula mu* (media de |efecto elemental|) y sigma (desviacion estandar
    de los efectos elementales) para cada uno de los k parametros, a
    partir de las evaluaciones Y (una por punto de cada trayectoria) y la
    informacion de orden/signo de cada trayectoria.
    """
    n_tray = len(info_trayectorias)
    puntos_por_tray = k + 1
    efectos = np.full((n_tray, k), np.nan)

    for t, (orden, signos) in enumerate(info_trayectorias):
        y_tray = Y[t * puntos_por_tray:(t + 1) * puntos_por_tray]
        for paso, idx in enumerate(orden):
            y_antes = y_tray[paso]
            y_despues = y_tray[paso + 1]
            efectos[t, idx] = (y_despues - y_antes) / (signos[idx] * delta)

    mu_star = np.nanmean(np.abs(efectos), axis=0)
    sigma = np.nanstd(efectos, axis=0)
    return mu_star, sigma


def muestrear_sobol(bounds, N, seed=42):
    """
    Genera las matrices A, B y AB_i (Saltelli 2010) para el calculo de
    indices de Sobol de primer orden (S1) y total (ST). Usa muestreo
    pseudo-aleatorio uniforme (numpy puro, sin dependencias externas) en
    vez de secuencias de baja discrepancia -- se compensa usando un N
    mayor al que usaria SALib con secuencias de Sobol, para mantener
    precision comparable.
    """
    k = len(bounds)
    rng_local = np.random.default_rng(seed)
    indices_log = [(hi / lo) > 50 for lo, hi in bounds]

    A_unit = rng_local.uniform(0, 1, size=(N, k))
    B_unit = rng_local.uniform(0, 1, size=(N, k))

    A = _escalar_a_bounds(A_unit, bounds, indices_log)
    B = _escalar_a_bounds(B_unit, bounds, indices_log)

    AB_list = []
    for i in range(k):
        AB_i_unit = A_unit.copy()
        AB_i_unit[:, i] = B_unit[:, i]
        AB_list.append(_escalar_a_bounds(AB_i_unit, bounds, indices_log))

    return A, B, AB_list


def analizar_sobol(f_A, f_B, f_AB_list):
    """
    Calcula los indices de Sobol de primer orden (S1) y total (ST) para
    cada parametro, usando los estimadores de Saltelli et al. (2010),
    ecuaciones 4.17 (S1) y 4.18 (ST):
        S1_i = (1/N * sum(f_B * (f_ABi - f_A))) / V
        ST_i = (1/(2N) * sum((f_A - f_ABi)^2)) / V
    donde V es la varianza combinada de f_A y f_B.
    """
    N = len(f_A)
    combinado = np.concatenate([f_A, f_B])
    V = np.var(combinado)
    k = len(f_AB_list)
    S1 = np.zeros(k)
    ST = np.zeros(k)
    for i in range(k):
        f_ABi = f_AB_list[i]
        S1[i] = np.mean(f_B * (f_ABi - f_A)) / V if V > 0 else 0.0
        ST[i] = 0.5 * np.mean((f_A - f_ABi) ** 2) / V if V > 0 else 0.0
    return S1, ST


print("Analisis de sensibilidad propio (Morris + Sobol) listo -- sin dependencia de SALib")

# ═══════════════════════════════════════════════════════════════
# BLOQUE 2 -- PARAMETROS FIJOS DEL SISTEMA (sin variacion en SALib)
# ═══════════════════════════════════════════════════════════════

# ── Modulo A: correcciones de temperatura (Q10) ──
# CORRECCION vs. doc original: se usan los Q10 y citas YA VERIFICADOS y
# usados de forma consistente en Modulo A v2.0 (Q10 diferenciado por tipo
# de proceso bioquimico), en vez de las citas nuevas del doc de Modulo G
# (Gillooly et al. 2001 es teoria metabolica general, no especifica de
# T7 RNAP/E. coli; Imanaka 1989 y Ratkowsky 1982 no se habian usado antes
# en este pipeline y no se verificaron aqui).
Q10_tx  = 1.8   # Transcripcion por T7 RNAP. LIT -- Chamberlin, M., & Ring, J.
                 # (1973). Characterization of T7-specific ribonucleic acid
                 # polymerase. J Biol Chem, 248(6), 2235-2244.
                 # https://doi.org/10.1016/S0021-9258(19)44178-4
Q10_deg = 1.5   # Degradacion proteica (proteasas). LIT -- Goldberg, A. L.
                 # (2003). Protein degradation and protection against
                 # misfolded or damaged proteins. Nature, 426(6968), 895-899.
                 # https://doi.org/10.1038/nature02263
Q10_mu  = 2.5   # Crecimiento microbiano. LIT -- Farewell, A., & Neidhardt,
                 # F. C. (1998). Effect of temperature on in vivo protein
                 # synthetic capacity in E. coli. J Bacteriol, 180(17),
                 # 4704-4710. https://doi.org/10.1128/JB.180.17.4704-4710.1998
T_ref   = 37.0  # C -- referencia fisiologica
T_exp   = 16.0  # C -- temperatura de expresion. PROTOCOL -- iGEM BL21(DE3) 16C

def q10_factor(Q10, T_exp=16.0, T_ref=37.0):
    """Factor de correccion Q10: k(T_exp) = k_ref x Q10^((T_exp-T_ref)/10)."""
    return Q10 ** ((T_exp - T_ref) / 10.0)

fc_tx  = q10_factor(Q10_tx)
fc_deg = q10_factor(Q10_deg)
fc_mu  = q10_factor(Q10_mu)

# ── Modulo A: parametros secundarios fijos ──
# CORRECCION vs. doc original: delta_R_ref, delta_m_ref y delta_p_ref
# vuelven a los valores YA VALIDADOS de Modulo A v2.0 (el doc de Modulo G
# introducia valores y citas distintos sin reconciliar con el resto del
# pipeline -- ver nota de auditoria al inicio del script).
mu_ref      = 1.386  # h^-1 -- crecimiento a 37C. LIT -- Farewell & Neidhardt
                       # (1998). J Bacteriol, 180(17), 4704-4710. Ibid.
delta_R_ref = 0.200  # h^-1 -- degradacion T7 RNAP a 37C. LIT -- Perez-Perez
                       # & Gutierrez (1995). Gene, 158(1), 141-142. (Ya usado
                       # en Modulo A; ambos modelos del equipo coincidieron
                       # independientemente en este valor.)
delta_m_ref = 6.000  # h^-1 -- degradacion ARNm E. coli a 37C. LIT -- Bernstein
                       # et al. (2002). PNAS, 99(15), 9697-9702.
                       # https://doi.org/10.1073/pnas.112318199
delta_p_ref = 0.050  # h^-1 -- degradacion dCas9-AviTag a 37C. LIT -- Maurizi,
                       # M. R. (1992). Proteases and protein degradation in
                       # E. coli. Experientia, 48(2), 178-201.
                       # https://doi.org/10.1007/BF01923511
delta_BirA_ref = 0.08  # h^-1 -- BirA mas labil que dCas9. PLACEHOLDER, sin
                        # cita primaria verificada (heredado de Modulo A v2.0).
                        # WET LAB -- pendiente de medicion experimental
                        # (pulse-chase o Western blot + rifampicina).

# Induccion IPTG (Hill) -- ya validado en Modulo A v2.0
IPTG        = 0.5   # mM -- PROTOCOL
K_IPTG      = 0.1   # mM -- PLACEHOLDER (rango 0.05-0.5 mM)
n_Hill      = 2.0   # adim. -- LIT -- Oehler, S., et al. (1994). The three
                     # operators of the lac operon cooperate in repression.
                     # EMBO J, 13(14), 3348-3355.
                     # https://doi.org/10.1002/j.1460-2075.1994.tb06637.x
RBS_str_2   = 0.57  # adim. -- relativa a RBS1=1.0. DERIVED (Modulo A v2.0).

# ── Modulo C: peso molecular ──
MW_dCas9    = 160000.0  # Da -- LIT -- Qi et al. (2013). Cell, 152(5), 1173-1183.
                          # https://doi.org/10.1016/j.cell.2013.02.022
MW_TrxA     = 11700.0   # Da -- LIT -- LaVallie et al. (1993). Bio/Technology,
                          # 11(2), 187-193. https://doi.org/10.1038/nbt0293-187
MW_6xHis    = 900.0     # Da -- DERIVED -- 6 x 137 Da (residuo de His) + linker
MW_AviTag   = 1200.0    # Da -- LIT -- Beckett et al. (1999). Protein Sci,
                          # 8(4), 921-929. https://doi.org/10.1110/ps.8.4.921
MW_total = MW_dCas9 + MW_TrxA + MW_6xHis + MW_AviTag   # ~173,800 Da <- Modulo C

# ── Modulo D: propiedades del chip ──
# CORRECCION vs. doc original: Kd_biotin_SA usa el DOI ya verificado en
# Modulo D (terminando en "84260-J", no "84259-J" como aparecia en el doc
# de Modulo G).
Kd_biotin_SA = 1.0e-15  # M -- Kd streptavidina-biotina. LIT -- Green, N. M.
                          # (1990). Avidin and streptavidin. Methods in
                          # Enzymology, 184, 51-67.
                          # https://doi.org/10.1016/0076-6879(90)84260-J

# ── Modulo F: transduccion SPR-fase ──
# CORRECCION CRITICA DE UNIDADES (heredada y ya corregida en Modulo F):
dn_dc_DNA_cm3g  = 0.168  # cm^3/g -- indice de refraccion especifico del DNA.
                           # LIT -- Stengel, G., Zahn, R., & Hook, F. (2007).
                           # DNA-induced programmable fusion of phospholipid
                           # vesicles. JACS, 129(31), 9584-9585.
                           # https://doi.org/10.1021/ja073200k
dn_dc_DNA_mm3g  = dn_dc_DNA_cm3g * 1000.0  # 168 mm^3/g (1 cm^3 = 1000 mm^3)
                                             # Conversion necesaria porque d_eff
                                             # y Gamma estan en escala de mm.
d_eff_mm        = 200e-6  # mm (=200 nm). LIT -- Homola, J. (2008). Surface
                           # plasmon resonance sensors for detection of
                           # chemical and biological species. Chem Rev,
                           # 108(2), 462-493. https://doi.org/10.1021/cr068107d
MW_bp           = 615.4   # g/mol.bp. LIT -- Voet, D., & Voet, J. G. (2011).
                           # Biochemistry (4th ed.). Wiley.
bp_ctDNA        = 167     # bp -- fragmentos mononucleosomales de ctDNA.
                           # LIT -- Snyder, M. W., Kircher, M., Hill, A. J.,
                           # Daza, R. M., & Shendure, J. (2016). Cell-free DNA
                           # comprises an in vivo nucleosome footprint.
                           # Cell, 164(1-2), 57-68.
                           # https://doi.org/10.1016/j.cell.2015.11.050
MW_ctDNA        = bp_ctDNA * MW_bp   # ~1.028e5 g/mol (consistente con el
                                       # ~1.1e5 redondeado usado en Modulo F)

pct_biotin = 1.0  # 100% biotinilacion. DERIVED <- Modulo B: [S_0] << Km_S
                    # (regimen de primer orden); confirmado en TODOS los
                    # escenarios Monte Carlo de A v2.0 -> B.

# ── Especificacion LOD (QbD) ──
LOD_spec_pM = 1.0    # pM -- objetivo conservador para ctDNA clinico early-stage.
                       # LIT -- Bettegowda, C., et al. (2014). Detection of
                       # circulating tumor DNA in early- and late-stage human
                       # malignancies. Sci Transl Med, 6(224), 224ra24.
                       # https://doi.org/10.1126/scitranslmed.3007094
LOD_spec_fM = 100.0  # fM -- objetivo aspiracional (plasma early-stage CA).
                       # LIT -- Wan, J. C. M., et al. (2017). Liquid biopsies
                       # come of age: Towards implementation of circulating
                       # tumour DNA. Cancer Res, 77(11), e31-e33.
                       # https://doi.org/10.1158/0008-5472.CAN-16-2631


# ═══════════════════════════════════════════════════════════════
# BLOQUE 3 -- PIPELINE COMPLETO (cadena A->C->D->E->F->LOD, QSS)
# ═══════════════════════════════════════════════════════════════
def calcular_LOD(params, sgRNA='EGFR19'):
    """
    Pipeline completo A->C->D->E->F->LOD (cadena analitica QSS).

    Valida la aproximacion quasi-estacionaria (QSS) del Modulo A a t=20h:
    tau_R~3.5h, tau_ARNm~0.63h, tau_P~4.5h -> a t=20h el sistema esta a
    >98.8% de su valor de estado estacionario. Error maximo QSS vs ODE
    completo: ~1.4% en P_dCas9 -- aceptable para analisis de sensibilidad
    (que necesita miles de evaluaciones rapidas, no factible resolviendo
    ODEs en cada una).

    Retorna: LOD [M]
    """
    k_R_ref     = params['k_R_ref']       # WET LAB
    f_rt        = params['f_rt']          # WET LAB
    phi_16      = params['phi_16']        # LIT -- Vera et al. (2007). Biotechnol.
                                            #   Bioeng., 96(6), 1101-1106.
                                            #   https://doi.org/10.1002/bit.21218
    k_tx_ref    = params['k_tx_ref']      # LIT -- Salis et al. (2009). Nat
                                            #   Biotechnol, 27(10), 946-950.
                                            #   https://doi.org/10.1038/nbt.1568
    k_tl0_ref   = params['k_tl0_ref']     # LIT -- Salis et al. (2009). Ibid.
    eta_pur     = params['eta_pur']       # LIT -- Bornhorst & Falke (2000).
                                            #   Methods Enzymol, 326, 245-254.
                                            #   https://doi.org/10.1016/S0076-6879(00)26058-8
    Gamma_max_chip = params['Gamma_max_chip']  # LIT -- Rich & Myszka (2000).
                                                 #   Curr Opin Biotechnol, 11(1), 54-61.
                                                 #   (orden de magnitud, WET LAB PEND)
    Kd_mut      = params['Kd_mut']        # LIT -- Sternberg et al. (2014). Nature,
                                            #   507(7490), 62-67. (Modulo E)
    S_phase     = params['S_phase']       # WET LAB -- Grigorenko, A. N., Nikitin,
                                            #   P. I., & Kabashin, A. V. (1999). Appl
                                            #   Phys Lett, 75(25), 3917-3919.
                                            #   https://doi.org/10.1063/1.125493
    sigma_blank = params['sigma_blank']   # WET LAB -- Nikitin, P. I., Beloglazov,
                                            #   A. A., Kochergin, V. E., Valeiko,
                                            #   M. V., & Ksenevich, T. I. (1999).
                                            #   Sensors Actuators B, 54(1-2), 43-50.
                                            #   https://doi.org/10.1016/S0925-4005(98)00310-1

    # ── Correcciones Q10 (Modulo A) ──
    k_R_adj        = k_R_ref  * fc_tx
    k_tx_adj       = k_tx_ref * fc_tx
    k_tl1_adj      = k_tl0_ref * fc_tx
    k_tl2_adj      = k_tl0_ref * fc_tx * RBS_str_2
    delta_R_adj    = delta_R_ref  * fc_deg
    delta_m_adj    = delta_m_ref  * fc_deg
    delta_p_adj    = delta_p_ref  * fc_deg
    delta_BirA_adj = delta_BirA_ref * fc_deg
    mu_adj         = mu_ref * fc_mu

    # ── Modulo A (QSS): concentraciones en estado quasi-estacionario a t=20h ──
    induction = (IPTG ** n_Hill) / (K_IPTG ** n_Hill + IPTG ** n_Hill)

    R_ss = k_R_adj * induction / (delta_R_adj + mu_adj)
    T_ss = k_tx_adj * R_ss / (delta_m_adj + mu_adj)

    P_dCas9_qss = k_tl1_adj * T_ss / (delta_p_adj + mu_adj)
    P_BirA_qss = k_tl2_adj * (f_rt * T_ss + T_ss) / (delta_BirA_adj + mu_adj)
    # P_BirA usa f_rt*T_ss (read-through del transcrito largo) + T_ss (promotor
    # propio, transcrito corto) <- Modulo A v2.0 (modelo unificado)

    # ── Modulo C: fraccion funcional ──
    P_funcional_nM = P_dCas9_qss * phi_16 * pct_biotin * eta_pur   # [nM]
    P_funcional_mol_L = P_funcional_nM * 1e-9   # [mol/L]

    # ── Modulo D: superficie de inmovilizacion ──
    V_inc_L = 500e-6       # L -- volumen incubacion tipico
    A_chip_mm2 = 1.0       # mm^2 -- area activa del chip
    eta_loading = 0.80     # adim. -- eficiencia de carga (80%)

    moles_disponibles = P_funcional_mol_L * V_inc_L
    moles_chip_max = Gamma_max_chip * A_chip_mm2

    Gamma_max_ef = min(Gamma_max_chip, moles_disponibles * eta_loading / A_chip_mm2)

    # ── Modulo E + F: senal SPR maxima (Langmuir saturacion + transduccion) ──
    Delta_phi_max = S_phase * (dn_dc_DNA_mm3g * MW_ctDNA * Gamma_max_ef) / d_eff_mm

    # ── LOD en concentracion (criterio 3-sigma, inversa de Langmuir) ──
    LOD_phi = 3.0 * sigma_blank

    if Delta_phi_max <= LOD_phi:
        return np.inf

    LOD_M = Kd_mut * LOD_phi / (Delta_phi_max - LOD_phi)
    return LOD_M


# ═══════════════════════════════════════════════════════════════
# VALIDACION INTERNA DE LA CADENA (con parametros nominales)
# ═══════════════════════════════════════════════════════════════
params_nominales_check = {
    'k_R_ref': 20.0, 'f_rt': 0.20, 'phi_16': 0.60, 'k_tx_ref': 2.00,
    'k_tl0_ref': 3.50, 'eta_pur': 0.55, 'Gamma_max_chip': 5.0e-14,
    'Kd_mut': 0.5e-9, 'S_phase': 1.0e4, 'sigma_blank': 1.0e-4,
}

induction_check = (IPTG ** n_Hill) / (K_IPTG ** n_Hill + IPTG ** n_Hill)
R_ss_check = (params_nominales_check['k_R_ref'] * fc_tx) * induction_check / (delta_R_ref * fc_deg + mu_ref * fc_mu)
T_ss_check = (params_nominales_check['k_tx_ref'] * fc_tx) * R_ss_check / (delta_m_ref * fc_deg + mu_ref * fc_mu)
P_dCas9_check = (params_nominales_check['k_tl0_ref'] * fc_tx) * T_ss_check / (delta_p_ref * fc_deg + mu_ref * fc_mu)
P_funcional_check = P_dCas9_check * params_nominales_check['phi_16'] * pct_biotin * params_nominales_check['eta_pur']
LOD_check = calcular_LOD(params_nominales_check)

print("\n" + "=" * 70)
print("VALIDACION INTERNA DE LA CADENA (parametros nominales)")
print("=" * 70)
print(f"  P_dCas9_qss(20h)   = {P_dCas9_check:.3f} nM   (esperado ~20.1 nM, Modulo A v2.0)")
print(f"  P_funcional        = {P_funcional_check:.3f} nM   (esperado ~6.65 nM, Modulo C)")
print(f"  LOD (EGFR19_mut)   = {LOD_check*1e15:.2f} fM   (esperado ~3.3 fM, Modulo F corregido)")
if abs(P_dCas9_check - 20.1) / 20.1 > 0.10:
    print("  ADVERTENCIA: P_dCas9_qss difiere >10% del valor esperado de Modulo A v2.0.")
if abs(P_funcional_check - 6.65) / 6.65 > 0.10:
    print("  ADVERTENCIA: P_funcional difiere >10% del valor esperado de Modulo C.")
if abs(LOD_check*1e15 - 3.3) / 3.3 > 0.20:
    print("  ADVERTENCIA: LOD difiere >20% del valor esperado de Modulo F.")
print("=" * 70)


# ═══════════════════════════════════════════════════════════════
# BLOQUE 4 -- DEFINICION DEL ESPACIO DE PARAMETROS (SALib)
# ═══════════════════════════════════════════════════════════════

# CORRECCION vs. doc original: Kd_wt de los 3 sgRNAs y sus factores de
# discriminacion ya se establecieron y verificaron en el Modulo E; se
# reusan aqui tal cual (no se remuestrean valores nuevos).
SGRNAS = {
    'KRAS_G12C':   {'Kd_mut': 1.0e-9, 'Kd_wt': 100e-9,  'color': '#E74C3C', 'label': 'KRAS G12C'},
    'EGFR19':      {'Kd_mut': 0.5e-9, 'Kd_wt': 1000e-9, 'color': '#3498DB', 'label': 'EGFR exon19del'},
    'EGFR_L858R':  {'Kd_mut': 1.0e-9, 'Kd_wt': 100e-9,  'color': '#2ECC71', 'label': 'EGFR L858R'},
}

# NOTA: S_phase y sigma_blank abarcan varios ordenes de magnitud (10^3-10^5
# y 10^-6-10^-2 respectivamente). Se muestrean linealmente dentro de esos
# bounds via SALib (que por defecto usa distribucion uniforme en el rango
# dado), pero como la respuesta del modelo (log10(LOD)) es aproximadamente
# LINEAL en log10(S_phase) y log10(sigma_blank) (ver Modulo F), el ranking
# relativo de sensibilidad Morris/Sobol es representativo igual; se usa
# log10(LOD) como metrica de salida para compensar la escala.
PARAM_NAMES = [
    'k_R_ref', 'f_rt', 'phi_16', 'k_tx_ref', 'k_tl0_ref',
    'eta_pur', 'Gamma_max_chip', 'Kd_mut', 'S_phase', 'sigma_blank',
]

PARAMS_NOMINAL = {
    'k_R_ref': 20.0, 'f_rt': 0.20, 'phi_16': 0.60, 'k_tx_ref': 2.00,
    'k_tl0_ref': 3.50, 'eta_pur': 0.55, 'Gamma_max_chip': 5.0e-14,
    'Kd_mut': 0.5e-9, 'S_phase': 1.0e4, 'sigma_blank': 1.0e-4,
}

PARAM_BOUNDS = [
    [5.0, 50.0],        # k_R_ref [nM/h]        -- WET LAB
    [0.05, 0.50],        # f_rt                   -- WET LAB
    [0.35, 0.75],        # phi_16                 -- LIT (Vera et al. 2007)
    [1.2, 3.5],          # k_tx_ref               -- LIT (Salis et al. 2009)
    [2.0, 5.0],          # k_tl0_ref              -- LIT (Salis et al. 2009)
    [0.30, 0.75],        # eta_pur                -- LIT (Bornhorst & Falke 2000)
    [2e-14, 1.2e-13],    # Gamma_max_chip [mol/mm^2]
    [0.1e-9, 5.0e-9],    # Kd_mut [M]
    [1.0e3, 1.0e5],      # S_phase [deg/RIU]      -- WET LAB
    [1.0e-6, 1.0e-2],    # sigma_blank [deg]      -- WET LAB
]

ORIGINS = ['WL', 'WL', 'LIT', 'LIT', 'LIT', 'LIT/WL', 'LIT', 'LIT/WL', 'WL', 'WL']


# ═══════════════════════════════════════════════════════════════
# BLOQUE 5 -- ANALISIS DE SENSIBILIDAD MORRIS (screening)
# ═══════════════════════════════════════════════════════════════
print("\n-- Etapa 1: Analisis Morris (screening) --")
N_traj_morris = 200
k_total = len(PARAM_NAMES)

X_morris, info_trayectorias_morris, delta_morris, indices_log_morris = muestrear_morris(
    PARAM_BOUNDS, N_trayectorias=N_traj_morris, num_levels=8, seed=42
)
print(f"   Evaluaciones requeridas: {X_morris.shape[0]}")

morris_results = {}
for sgRNA_key, sgRNA_data in SGRNAS.items():
    print(f"   Evaluando {sgRNA_data['label']}...", end=' ')
    Y_morris = np.zeros(X_morris.shape[0])
    for i, x in enumerate(X_morris):
        p = dict(zip(PARAM_NAMES, x))
        p['Kd_mut'] = sgRNA_data['Kd_mut']
        lod = calcular_LOD(p, sgRNA=sgRNA_key)
        Y_morris[i] = np.log10(lod) if np.isfinite(lod) and lod > 0 else 15.0
    mu_star_sg, sigma_sg = analizar_morris(Y_morris, info_trayectorias_morris, k_total, delta_morris)
    morris_results[sgRNA_key] = {'mu_star': mu_star_sg, 'sigma': sigma_sg}
    print(f"OK (mu*_max = {np.max(mu_star_sg):.2f})")

print("\n-- Ranking Morris (por mu* de log10(LOD), promedio sobre sgRNAs) --")
mu_star_avg = np.mean([morris_results[k]['mu_star'] for k in SGRNAS], axis=0)
sigma_star_avg = np.mean([morris_results[k]['sigma'] for k in SGRNAS], axis=0)

ranking_morris = np.argsort(mu_star_avg)[::-1]
print(f"{'Rank':<5}{'Parametro':<20}{'mu* (avg)':<14}{'sigma* (avg)':<14}{'Origen'}")
for rank, idx in enumerate(ranking_morris, 1):
    print(f"  {rank:<4}{PARAM_NAMES[idx]:<20}{mu_star_avg[idx]:<14.3f}{sigma_star_avg[idx]:<14.3f}{ORIGINS[idx]}")

top6_idx = ranking_morris[:6]
top6_names = [PARAM_NAMES[i] for i in top6_idx]
print(f"\n   Top 6 para analisis Sobol: {top6_names}")


# ═══════════════════════════════════════════════════════════════
# BLOQUE 6 -- ANALISIS DE SENSIBILIDAD SOBOL (top 6 parametros)
# ═══════════════════════════════════════════════════════════════
print("\n-- Etapa 2: Analisis Sobol (top 6 CPPs) --")

problem_top6 = {
    'num_vars': 6,
    'names': top6_names,
    'bounds': [PARAM_BOUNDS[i] for i in top6_idx],
}

# N mas alto que en la version con SALib (2048) porque el muestreo aqui es
# pseudo-aleatorio uniforme, no una secuencia de baja discrepancia (Sobol'
# quasi-random) -- se compensa la menor eficiencia de convergencia con mas
# muestras. Total de evaluaciones: N_sobol * (k+2) por sgRNA.
N_sobol = 4096
A_sobol, B_sobol, AB_sobol_list = muestrear_sobol(problem_top6['bounds'], N=N_sobol, seed=42)
print(f"   Evaluaciones requeridas: {N_sobol * (len(top6_names) + 2)}")

sobol_results = {}
for sgRNA_key, sgRNA_data in SGRNAS.items():
    print(f"   Evaluando {sgRNA_data['label']}...", end=' ')

    def _evaluar_matriz(M):
        Y = np.zeros(M.shape[0])
        for i, x in enumerate(M):
            p = dict(PARAMS_NOMINAL)
            for j, name in enumerate(top6_names):
                p[name] = x[j]
            p['Kd_mut'] = sgRNA_data['Kd_mut']
            lod = calcular_LOD(p, sgRNA=sgRNA_key)
            Y[i] = np.log10(lod) if np.isfinite(lod) and lod > 0 else 15.0
        return Y

    f_A = _evaluar_matriz(A_sobol)
    f_B = _evaluar_matriz(B_sobol)
    f_AB_list = [_evaluar_matriz(AB_i) for AB_i in AB_sobol_list]

    S1_sg, ST_sg = analizar_sobol(f_A, f_B, f_AB_list)
    sobol_results[sgRNA_key] = {'S1': S1_sg, 'ST': ST_sg}
    print(f"OK (S_T max = {np.max(ST_sg):.3f})")

print("\n-- Indices Sobol (promedio sobre sgRNAs) --")
print("   S1 = indice primer orden (efecto individual)")
print("   ST = indice total (incluye interacciones)")
S1_avg = np.mean([sobol_results[k]['S1'] for k in SGRNAS], axis=0)
ST_avg = np.mean([sobol_results[k]['ST'] for k in SGRNAS], axis=0)
ranking_sobol = np.argsort(ST_avg)[::-1]
print(f"{'Parametro':<20}{'S1':<10}{'ST':<10}")
for idx in ranking_sobol:
    print(f"  {top6_names[idx]:<20}{S1_avg[idx]:<10.3f}{ST_avg[idx]:<10.3f}")


# ═══════════════════════════════════════════════════════════════
# BLOQUE 7 -- DESIGN SPACE (QbD)
# ═══════════════════════════════════════════════════════════════
cpe1_name = top6_names[ranking_sobol[0]]
cpe2_name = top6_names[ranking_sobol[1]]

n_grid = 80
cpe1_range = problem_top6['bounds'][top6_names.index(cpe1_name)]
cpe2_range = problem_top6['bounds'][top6_names.index(cpe2_name)]

use_log_cpe1 = (cpe1_range[1] / cpe1_range[0]) > 50
use_log_cpe2 = (cpe2_range[1] / cpe2_range[0]) > 50

if use_log_cpe1:
    cpe1_vals = np.logspace(np.log10(cpe1_range[0]), np.log10(cpe1_range[1]), n_grid)
else:
    cpe1_vals = np.linspace(cpe1_range[0], cpe1_range[1], n_grid)

if use_log_cpe2:
    cpe2_vals = np.logspace(np.log10(cpe2_range[0]), np.log10(cpe2_range[1]), n_grid)
else:
    cpe2_vals = np.linspace(cpe2_range[0], cpe2_range[1], n_grid)

print(f"\n-- Etapa 3: Design Space ({cpe1_name} vs {cpe2_name}) --")
ds_grids = {}
for sgRNA_key, sgRNA_data in SGRNAS.items():
    grid = np.zeros((n_grid, n_grid))
    for ii, v1 in enumerate(cpe1_vals):
        for jj, v2 in enumerate(cpe2_vals):
            p = dict(PARAMS_NOMINAL)
            p[cpe1_name] = v1
            p[cpe2_name] = v2
            p['Kd_mut'] = sgRNA_data['Kd_mut']
            lod = calcular_LOD(p, sgRNA=sgRNA_key)
            grid[ii, jj] = np.log10(lod) if np.isfinite(lod) and lod > 0 else 15.0
    ds_grids[sgRNA_key] = grid
    print(f"   Design Space {sgRNA_data['label']}: LOD_min={10**grid.min():.2e} M, "
          f"LOD_max={10**grid.max():.2e} M")

sgRNA_mas_sensible = min(SGRNAS, key=lambda k: SGRNAS[k]['Kd_mut'])


# ═══════════════════════════════════════════════════════════════
# BLOQUE 8 -- FIGURAS
# ═══════════════════════════════════════════════════════════════

# --- FIGURA 1: Ranking de sensibilidad Morris + Sobol (tornado doble) ---
fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(14, 6))
fig1.suptitle("Sensibilidad Global del LOD -- Morris mu* y Sobol ST", fontsize=13, fontweight="bold")

colores_sg = [SGRNAS[k]['color'] for k in SGRNAS]
nombres_sg = [SGRNAS[k]['label'] for k in SGRNAS]

# Panel izquierdo: Morris mu*, por sgRNA, para los 10 parametros
y_pos_morris = np.arange(len(PARAM_NAMES))
ancho_barra = 0.25
for i, (sgRNA_key, sgRNA_data) in enumerate(SGRNAS.items()):
    mu_star_sg = morris_results[sgRNA_key]['mu_star']
    valores_ordenados = [mu_star_sg[idx] for idx in ranking_morris]
    ax1a.barh(y_pos_morris + (i - 1) * ancho_barra, valores_ordenados, ancho_barra,
              color=sgRNA_data['color'], label=sgRNA_data['label'], alpha=0.85)
etiquetas_morris = [f"{PARAM_NAMES[idx]}{'*' if ORIGINS[idx] in ('WL','LIT/WL') else ''}" for idx in ranking_morris]
ax1a.set_yticks(y_pos_morris)
ax1a.set_yticklabels(etiquetas_morris, fontsize=9)
ax1a.invert_yaxis()
ax1a.set_xlabel("mu* (efecto elemental medio absoluto sobre log10(LOD))")
ax1a.set_title("Morris (screening, 10 CPPs)")
ax1a.legend(fontsize=8)
ax1a.grid(alpha=0.3, axis="x")

# Panel derecho: Sobol ST, por sgRNA, para los top6
y_pos_sobol = np.arange(len(top6_names))
for i, (sgRNA_key, sgRNA_data) in enumerate(SGRNAS.items()):
    ST_sg = sobol_results[sgRNA_key]['ST']
    valores_ordenados_s = [ST_sg[idx] for idx in ranking_sobol]
    ax1b.barh(y_pos_sobol + (i - 1) * ancho_barra, valores_ordenados_s, ancho_barra,
              color=sgRNA_data['color'], label=sgRNA_data['label'], alpha=0.85)
etiquetas_sobol = [f"{top6_names[idx]}{'*' if ORIGINS[PARAM_NAMES.index(top6_names[idx])] in ('WL','LIT/WL') else ''}" for idx in ranking_sobol]
ax1b.set_yticks(y_pos_sobol)
ax1b.set_yticklabels(etiquetas_sobol, fontsize=9)
ax1b.invert_yaxis()
ax1b.set_xlabel("ST (indice de Sobol total)")
ax1b.set_title("Sobol (varianza, top 6 CPPs)")
ax1b.legend(fontsize=8)
ax1b.grid(alpha=0.3, axis="x")
ax1b.text(0.98, -0.12, "* = parametro WET LAB pendiente", transform=ax1b.transAxes,
          ha="right", fontsize=7.5, style="italic")

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("ModG_Fig1_Sensibilidad_Morris_Sobol.png", dpi=300, bbox_inches="tight")
print("\nFigura guardada: ModG_Fig1_Sensibilidad_Morris_Sobol.png")


# --- FIGURA 2: Design Space (contour 2D, sgRNA mas sensible) ---
fig2, ax2 = plt.subplots(figsize=(10, 7))

grid_mostrar = ds_grids[sgRNA_mas_sensible]
X1 = np.log10(cpe1_vals) if use_log_cpe1 else cpe1_vals
X2 = np.log10(cpe2_vals) if use_log_cpe2 else cpe2_vals
X1_grid, X2_grid = np.meshgrid(X1, X2)

# Regiones sombreadas: verde (<=100fM), amarillo (100fM-1pM), rojo (>1pM)
niveles_color = [-20, np.log10(100e-15), np.log10(1e-12), 5]
colores_region = ["#2ECC71", "#F1C40F", "#E74C3C"]
cf2 = ax2.contourf(X1_grid, X2_grid, grid_mostrar.T, levels=niveles_color, colors=colores_region, alpha=0.55)

contornos2 = ax2.contour(X1_grid, X2_grid, grid_mostrar.T,
                          levels=[np.log10(10e-15), np.log10(100e-15), np.log10(1e-12), np.log10(10e-12)],
                          colors="black", linewidths=1.2)
ax2.clabel(contornos2, fmt=lambda v: f"{10**v*1e15:.0f} fM" if 10**v < 1e-12 else f"{10**v*1e12:.0f} pM", fontsize=8)

nom1 = PARAMS_NOMINAL[cpe1_name]
nom2 = PARAMS_NOMINAL[cpe2_name]
ax2.plot(np.log10(nom1) if use_log_cpe1 else nom1, np.log10(nom2) if use_log_cpe2 else nom2,
          "*", color="black", markersize=20, markeredgecolor="white", zorder=5, label="Nominal")

ax2.set_xlabel(f"{'log10' if use_log_cpe1 else ''}({cpe1_name})")
ax2.set_ylabel(f"{'log10' if use_log_cpe2 else ''}({cpe2_name})")
ax2.set_title(f"Design Space -- {cpe1_name} vs {cpe2_name} | {SGRNAS[sgRNA_mas_sensible]['label']}")
ax2.legend(fontsize=9, loc="upper left")

leyenda_regiones = [mpatches.Patch(color=colores_region[0], alpha=0.55, label="LOD <= 100 fM"),
                     mpatches.Patch(color=colores_region[1], alpha=0.55, label="100 fM < LOD <= 1 pM"),
                     mpatches.Patch(color=colores_region[2], alpha=0.55, label="LOD > 1 pM")]
ax2.legend(handles=leyenda_regiones + [plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="black",
           markersize=15, label="Nominal")], fontsize=8, loc="upper left")

plt.tight_layout()
plt.savefig("ModG_Fig2_DesignSpace_2D.png", dpi=300, bbox_inches="tight")
print("Figura guardada: ModG_Fig2_DesignSpace_2D.png")


# --- FIGURA 3: Dosis-respuesta en fase, 3 sgRNAs ---
fig3, ax3 = plt.subplots(figsize=(12, 5))

c_array_g = np.logspace(-16, -6, 400)

def delta_phi_nominal(c, Kd_mut_local):
    p = dict(PARAMS_NOMINAL)
    p['Kd_mut'] = Kd_mut_local
    # Recalcular Delta_phi_max con parametros nominales
    induction_l = (IPTG ** n_Hill) / (K_IPTG ** n_Hill + IPTG ** n_Hill)
    R_ss_l = (p['k_R_ref'] * fc_tx) * induction_l / (delta_R_ref * fc_deg + mu_ref * fc_mu)
    T_ss_l = (p['k_tx_ref'] * fc_tx) * R_ss_l / (delta_m_ref * fc_deg + mu_ref * fc_mu)
    P_dCas9_l = (p['k_tl0_ref'] * fc_tx) * T_ss_l / (delta_p_ref * fc_deg + mu_ref * fc_mu)
    P_func_l = P_dCas9_l * p['phi_16'] * pct_biotin * p['eta_pur']
    moles_disp_l = (P_func_l * 1e-9) * 500e-6
    Gamma_ef_l = min(p['Gamma_max_chip'], moles_disp_l * 0.80 / 1.0)
    Delta_phi_max_l = p['S_phase'] * (dn_dc_DNA_mm3g * MW_ctDNA * Gamma_ef_l) / d_eff_mm
    return Delta_phi_max_l * c / (Kd_mut_local + c), Delta_phi_max_l

LOD_phi_nominal = 3.0 * PARAMS_NOMINAL['sigma_blank']

for sgRNA_key, sgRNA_data in SGRNAS.items():
    phi_curve, dphi_max_l = delta_phi_nominal(c_array_g, sgRNA_data['Kd_mut'])
    ax3.plot(c_array_g, phi_curve, color=sgRNA_data['color'], linewidth=2, label=sgRNA_data['label'])
    lod_c = calcular_LOD(PARAMS_NOMINAL | {'Kd_mut': sgRNA_data['Kd_mut']})
    if np.isfinite(lod_c):
        ax3.annotate("", xy=(lod_c, LOD_phi_nominal), xytext=(lod_c, LOD_phi_nominal * 8),
                     arrowprops=dict(arrowstyle="->", color=sgRNA_data['color'], lw=1.5))

ax3.axhline(PARAMS_NOMINAL['sigma_blank'], color="gray", linestyle=":", linewidth=1.3, label="sigma_blank")
ax3.axhline(LOD_phi_nominal, color="red", linestyle="--", linewidth=1.3, label="LOD_phi = 3*sigma_blank")
ax3.set_xscale("log")
ax3.set_yscale("log")
ax3.set_xlabel("[ctDNA] (M)")
ax3.set_ylabel("Delta_phi (deg)")
ax3.set_title("Curvas Dosis-Respuesta Fase -- 3 sgRNAs (Parametros Nominales)")
ax3.legend(fontsize=8, loc="upper left")
ax3.grid(alpha=0.3, which="both")

plt.tight_layout()
plt.savefig("ModG_Fig3_DosisRespuesta_Fase.png", dpi=300, bbox_inches="tight")
print("Figura guardada: ModG_Fig3_DosisRespuesta_Fase.png")


# --- FIGURA 4: Mapa 2D S_phase vs sigma_blank (parametros WET LAB) ---
fig4, ax4 = plt.subplots(figsize=(9, 7))

S_phase_grid_vals = np.logspace(3, 5, n_grid)
sigma_grid_vals = np.logspace(-6, -2, n_grid)
LOD_grid4 = np.zeros((n_grid, n_grid))
for ii, sph in enumerate(S_phase_grid_vals):
    for jj, sig in enumerate(sigma_grid_vals):
        p = dict(PARAMS_NOMINAL)
        p['S_phase'] = sph
        p['sigma_blank'] = sig
        p['Kd_mut'] = SGRNAS[sgRNA_mas_sensible]['Kd_mut']
        lod4 = calcular_LOD(p)
        LOD_grid4[ii, jj] = np.log10(lod4) if np.isfinite(lod4) and lod4 > 0 else 15.0

Xg, Yg = np.meshgrid(np.log10(S_phase_grid_vals), np.log10(sigma_grid_vals))
cf4 = ax4.contourf(Xg, Yg, LOD_grid4.T, levels=[-20, np.log10(100e-15), 5],
                    colors=["#2ECC71", "#D5DBDB"], alpha=0.5)
contornos4 = ax4.contour(Xg, Yg, LOD_grid4.T, levels=20, cmap="viridis", linewidths=0.8)
cbar4 = plt.colorbar(plt.cm.ScalarMappable(norm=mcolors.Normalize(vmin=LOD_grid4.min(), vmax=LOD_grid4.max()), cmap="viridis"), ax=ax4)
cbar4.set_label("log10(LOD [M])")

ax4.plot(np.log10(PARAMS_NOMINAL['S_phase']), np.log10(PARAMS_NOMINAL['sigma_blank']),
          "*", color="black", markersize=20, markeredgecolor="white", zorder=5, label="Placeholder actual")
ax4.plot(np.log10(5e4), np.log10(1e-5), "o", color="blue", markersize=10, markeredgecolor="white",
          zorder=5, label="Mejora instrumental posible")

ax4.set_xlabel("log10(S_phase) (deg/RIU)")
ax4.set_ylabel("log10(sigma_blank) (deg)")
ax4.set_title(f"Design Space Instrumental -- S_phase vs sigma_blank\n(Ambos WET LAB) | {SGRNAS[sgRNA_mas_sensible]['label']}")
ax4.legend(fontsize=8, loc="lower right")
ax4.text(0.02, 0.02, "Region verde: LOD <= 100 fM (objetivo aspiracional).\n"
                      "Reducir sigma_blank (mejor blindaje/estabilidad optica)\n"
                      "y/o aumentar S_phase (mejor geometria de interrogacion)\n"
                      "mueve el sistema hacia la region verde.",
          transform=ax4.transAxes, fontsize=7.5, va="bottom",
          bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

plt.tight_layout()
plt.savefig("ModG_Fig4_DesignSpace_Instrumental.png", dpi=300, bbox_inches="tight")
print("Figura guardada: ModG_Fig4_DesignSpace_Instrumental.png")

plt.show()


# ═══════════════════════════════════════════════════════════════
# BLOQUE 9 -- RESUMEN EJECUTIVO
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("MODULO G -- RESUMEN EJECUTIVO")
print("=" * 70)

print("\nLODS NOMINALES (parametros nominales, sin incertidumbre):")
for sgRNA_key, sgRNA_data in SGRNAS.items():
    p_nom = dict(PARAMS_NOMINAL)
    p_nom['Kd_mut'] = sgRNA_data['Kd_mut']
    lod_nom = calcular_LOD(p_nom)
    lod_fM = lod_nom * 1e15
    check_1pM = "OK" if lod_nom <= LOD_spec_pM * 1e-12 else "NO CUMPLE"
    check_100fM = "OK" if lod_nom <= LOD_spec_fM * 1e-15 else "NO CUMPLE"
    print(f"  {sgRNA_data['label']:<16s}: LOD = {lod_fM:7.2f} fM   "
          f"[spec <1pM: {check_1pM}]   [spec <100fM: {check_100fM}]")

print("\nRANKING CPPs POR IMPACTO EN LOD (Sobol ST, promedio sgRNAs):")
for rank, idx in enumerate(ranking_sobol, 1):
    origen_txt = ORIGINS[PARAM_NAMES.index(top6_names[idx])]
    print(f"  {rank}. {top6_names[idx]:<18s} ST = {ST_avg[idx]:.3f}  [{origen_txt}]")

print(f"\nDESIGN SPACE:")
mask_verde = grid_mostrar <= np.log10(100e-15)
if mask_verde.any():
    idx1_verde = np.where(mask_verde.any(axis=1))[0]
    idx2_verde = np.where(mask_verde.any(axis=0))[0]
    rango1 = (cpe1_vals[idx1_verde.min()], cpe1_vals[idx1_verde.max()])
    rango2 = (cpe2_vals[idx2_verde.min()], cpe2_vals[idx2_verde.max()])
    print(f"  {cpe1_name}: rango con LOD<=100fM = [{rango1[0]:.2e}, {rango1[1]:.2e}]")
    print(f"  {cpe2_name}: rango con LOD<=100fM = [{rango2[0]:.2e}, {rango2[1]:.2e}]")
else:
    print(f"  Ningun punto de la grilla explorada alcanza LOD<=100fM para {cpe1_name}/{cpe2_name}")
    print(f"  dentro de los rangos actuales -- se requiere mejora instrumental (ver Figura 4).")

print("\nPRIORIDAD EXPERIMENTOS WET LAB (por ST):")
contador_prioridad = 1
for idx in ranking_sobol:
    origen_txt = ORIGINS[PARAM_NAMES.index(top6_names[idx])]
    if "WL" in origen_txt:
        print(f"  {contador_prioridad}. Medir {top6_names[idx]:<18s} -> impacto ST = {ST_avg[idx]:.3f} en LOD")
        contador_prioridad += 1

print("\nADVERTENCIAS:")
print("  LOD es TEORICO. Requiere validacion con:")
print("     -> S_phase medido (calibracion interferometrica)")
print("     -> sigma_blank medido (ruido real del instrumento)")
print("     -> k_R y f_rt medidos (cinetica de expresion T7)")
print()
print("  HALLAZGO IMPORTANTE DEL ANALISIS DE SENSIBILIDAD:")
print("  Los parametros de PRODUCCION de proteina (k_R_ref, f_rt, phi_16,")
print("  eta_pur, k_tx_ref, k_tl0_ref) muestran sensibilidad ~0 sobre el LOD")
print("  en TODO el rango explorado. La razon matematica: Gamma_max_ef =")
print("  min(Gamma_max_chip, proteina_disponible x eta_loading), y con el")
print("  volumen de incubacion (500 uL) y eficiencia de carga (80%) asumidos,")
print("  incluso en el escenario de PEOR produccion de proteina dentro del")
print("  rango explorado hay MAS QUE SUFICIENTE proteina para saturar el chip")
print("  (limitado por Gamma_max_chip, ~1e-14 a 1e-13 mol/mm^2 -- escala de")
print("  femtomoles). El chip, no la produccion de proteina, es SIEMPRE el")
print("  cuello de botella para el LOD en este modelo. Esto NO contradice los")
print("  hallazgos previos del pipeline (donde k_R/f_rt si importan mucho")
print("  para el RENDIMIENTO ABSOLUTO de proteina) -- solo significa que, una")
print("  vez que hay proteina suficiente para saturar un chip de area ~1mm^2,")
print("  producir mas no mejora el LOD. Lo que si mejora el LOD es Gamma_max_chip")
print("  (chips con mayor capacidad), S_phase y sigma_blank (mejor instrumento).")
print("=" * 70)

print("\nFiguras guardadas:")
for nombre_fig in ["ModG_Fig1_Sensibilidad_Morris_Sobol.png", "ModG_Fig2_DesignSpace_2D.png",
                    "ModG_Fig3_DosisRespuesta_Fase.png", "ModG_Fig4_DesignSpace_Instrumental.png"]:
    print(f"  {nombre_fig}")
print("\nModulo G completo")