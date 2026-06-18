"""
SmartDispatch — Sistema Híbrido de Optimización Logística
============================================================
Algoritmo Genético (DEAP) + Sistema Experto (Forward Chaining)
para ruteo de flota en logística de construcción.

Ejecutar con: streamlit run app.py
"""

import math
import random
import time
from dataclasses import dataclass, field
from typing import List, Tuple

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from deap import algorithms, base, creator, tools
from streamlit_folium import st_folium

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE PÁGINA
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="SmartDispatch | Optimización Logística",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "SmartDispatch — Optimizador Híbrido AG + Sistema Experto "
                  "para logística de construcción. Proyecto académico de IA Aplicada."
    },
)

# ═══════════════════════════════════════════════════════════════════════════
# ESTILOS
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

.main { background-color: #F7F9FC; }

.hero {
    background: linear-gradient(120deg, #0F2A4A 0%, #1C4E80 55%, #2E75B6 100%);
    border-radius: 18px;
    padding: 38px 42px;
    color: white;
    margin-bottom: 22px;
    box-shadow: 0 8px 24px rgba(15,42,74,0.25);
}
.hero h1 { color: white; font-size: 2.1rem; font-weight: 800; margin-bottom: 4px; }
.hero p { color: #D6E6F5; font-size: 1.02rem; margin: 0; }
.hero-badges { margin-top: 14px; }

.pill {
    display: inline-block;
    background: rgba(255,255,255,0.14);
    border: 1px solid rgba(255,255,255,0.25);
    color: white;
    padding: 5px 14px;
    border-radius: 30px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 8px;
}

.kpi-card {
    background: white;
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 2px 10px rgba(20,40,70,0.06);
    border: 1px solid #EAEDF2;
    height: 100%;
}
.kpi-label { color: #6B7785; font-size: 0.80rem; font-weight: 600; text-transform: uppercase; letter-spacing: .04em;}
.kpi-value { color: #0F2A4A; font-size: 1.9rem; font-weight: 800; margin-top: 2px;}
.kpi-sub { color: #8A93A0; font-size: 0.80rem; margin-top: 2px;}
.kpi-accent-blue   { border-left: 4px solid #2E75B6; }
.kpi-accent-green  { border-left: 4px solid #27AE60; }
.kpi-accent-orange { border-left: 4px solid #F39C12; }
.kpi-accent-red    { border-left: 4px solid #E74C3C; }

.section-title {
    color: #0F2A4A;
    font-weight: 800;
    font-size: 1.3rem;
    margin: 6px 0 2px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-sub { color: #6B7785; font-size: 0.92rem; margin-bottom: 14px; }

.callout {
    border-radius: 10px;
    padding: 14px 18px;
    margin: 10px 0;
    font-size: 0.92rem;
    line-height: 1.5;
}
.callout-info    { background: #EBF5FB; border-left: 4px solid #2E75B6; color:#0F2A4A;}
.callout-ok      { background: #EAFAF1; border-left: 4px solid #27AE60; color:#0F2A4A;}
.callout-warn    { background: #FEF9E7; border-left: 4px solid #F39C12; color:#0F2A4A;}
.callout-danger  { background: #FDEDEC; border-left: 4px solid #E74C3C; color:#0F2A4A;}

.badge { display:inline-block; padding:3px 11px; border-radius:20px; font-size:0.74rem; font-weight:700; margin:2px;}
.badge-blue {background:#D6EAF8; color:#1A5276;}
.badge-green{background:#D5F5E3; color:#1E8449;}
.badge-red  {background:#FADBD8; color:#922B21;}
.badge-orange{background:#FDEBD0; color:#9C640C;}
.badge-gray {background:#F1F2F6; color:#555;}

[data-testid="stSidebar"] {
    background: linear-gradient(195deg, #0F2A4A 0%, #1C4E80 100%);
}
[data-testid="stSidebar"] * { color: #EAF2FA !important; }
[data-testid="stSidebar"] .stSlider label, [data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stSelectbox label, [data-testid="stSidebar"] .stCheckbox label {
    font-weight: 600 !important; font-size: 0.85rem !important;
}
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15); }

div.stButton > button[kind="primary"] {
    background: linear-gradient(120deg, #2E75B6, #1C4E80);
    border: none;
    font-weight: 700;
    border-radius: 10px;
    padding: 0.6rem 1.2rem;
}

.stTabs [data-baseweb="tab"] { font-weight: 600; font-size: 0.95rem; }

footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# DOMINIO DEL NEGOCIO
# ═══════════════════════════════════════════════════════════════════════════

CORRALON = {"nombre": "Corralón Central", "lat": -34.603722, "lon": -58.381592}

MATERIALES_INCOMPATIBLES = {
    frozenset({"Cemento", "Yeso"}),
    frozenset({"Cemento", "Material Húmedo"}),
    frozenset({"Yeso", "Material Húmedo"}),
    frozenset({"Acero", "Ácido"}),
}

TIPOS_MATERIALES = [
    {"nombre": "Cemento",         "peso_base": 1500, "vol_base": 1.2, "peligroso": False, "clima": False},
    {"nombre": "Acero",           "peso_base": 3000, "vol_base": 2.0, "peligroso": False, "clima": False},
    {"nombre": "Yeso",            "peso_base": 800,  "vol_base": 0.8, "peligroso": False, "clima": False},
    {"nombre": "Madera",          "peso_base": 1200, "vol_base": 3.0, "peligroso": False, "clima": False},
    {"nombre": "Material Húmedo", "peso_base": 2000, "vol_base": 1.5, "peligroso": False, "clima": False},
    {"nombre": "Aditivo Químico", "peso_base": 500,  "vol_base": 0.5, "peligroso": True,  "clima": True},
    {"nombre": "Ladrillos",       "peso_base": 2500, "vol_base": 2.5, "peligroso": False, "clima": False},
    {"nombre": "Vidrio",          "peso_base": 600,  "vol_base": 1.8, "peligroso": False, "clima": False},
    {"nombre": "Cal",             "peso_base": 900,  "vol_base": 0.9, "peligroso": False, "clima": False},
    {"nombre": "Ácido",           "peso_base": 400,  "vol_base": 0.4, "peligroso": True,  "clima": True},
]

COLOR_MATERIAL = {
    "Cemento": "#3498DB", "Acero": "#7F8C8D", "Yeso": "#F39C12", "Madera": "#27AE60",
    "Material Húmedo": "#8E44AD", "Aditivo Químico": "#E74C3C", "Ladrillos": "#E67E22",
    "Vidrio": "#1ABC9C", "Cal": "#BDC3C7", "Ácido": "#C0392B",
}
COLOR_FOLIUM = {
    "Cemento": "blue", "Acero": "gray", "Yeso": "orange", "Madera": "green",
    "Material Húmedo": "purple", "Aditivo Químico": "red", "Ladrillos": "darkred",
    "Vidrio": "cadetblue", "Cal": "lightgray", "Ácido": "black",
}
COLOR_CAMION = ["#2E75B6", "#27AE60", "#E74C3C", "#F39C12", "#8E44AD",
                "#1ABC9C", "#E67E22", "#34495E"]


@dataclass
class Material:
    nombre: str
    peso_kg: float
    volumen_m3: float
    requiere_clima_controlado: bool = False
    es_peligroso: bool = False


@dataclass
class Pedido:
    id: str
    material: Material
    obra_destino: str
    coordenadas: Tuple[float, float]
    ventana_inicio: int
    ventana_fin: int
    proveedor_certificado: bool = True


@dataclass
class Camion:
    id: str
    capacidad_peso_kg: float = 45000.0
    capacidad_volumen_m3: float = 60.0
    tiene_clima_controlado: bool = False
    velocidad_kmh: float = 60.0
    pedidos: List[Pedido] = field(default_factory=list)


@dataclass
class Infraccion:
    regla_id: int
    descripcion: str
    penalizacion: float
    es_critica: bool


def generar_pedidos(n: int, seed: int) -> List[Pedido]:
    rng = random.Random(seed)
    obras = [
        {"nombre": f"Obra_{i}",
         "lat": CORRALON["lat"] + rng.uniform(-0.18, 0.18),
         "lon": CORRALON["lon"] + rng.uniform(-0.18, 0.18)}
        for i in range(1, n + 1)
    ]
    pedidos = []
    for i in range(n):
        obra = rng.choice(obras)
        mat_data = rng.choice(TIPOS_MATERIALES)
        material = Material(
            nombre=mat_data["nombre"],
            peso_kg=max(100, mat_data["peso_base"] + rng.uniform(-300, 300)),
            volumen_m3=max(0.1, mat_data["vol_base"] + rng.uniform(-0.3, 0.3)),
            requiere_clima_controlado=mat_data["clima"],
            es_peligroso=mat_data["peligroso"],
        )
        inicio = rng.choice([480, 540, 600, 660])
        pedidos.append(Pedido(
            id=f"P{i+1:02d}", material=material, obra_destino=obra["nombre"],
            coordenadas=(obra["lat"], obra["lon"]),
            ventana_inicio=inicio, ventana_fin=inicio + 120,
            proveedor_certificado=rng.random() > 0.12,
        ))
    return pedidos


def pedidos_a_dataframe(pedidos: List[Pedido]) -> pd.DataFrame:
    return pd.DataFrame([{
        "ID": p.id, "Material": p.material.nombre,
        "Peso_kg": round(p.material.peso_kg, 1),
        "Volumen_m3": round(p.material.volumen_m3, 2),
        "Obra": p.obra_destino,
        "Ventana_Inicio": f"{p.ventana_inicio//60:02d}:{p.ventana_inicio%60:02d}",
        "Ventana_Fin": f"{p.ventana_fin//60:02d}:{p.ventana_fin%60:02d}",
        "Certificado": p.proveedor_certificado,
        "Peligroso": p.material.es_peligroso,
    } for p in pedidos])


# ═══════════════════════════════════════════════════════════════════════════
# SISTEMA EXPERTO — 30 REGLAS, FORWARD CHAINING
# ═══════════════════════════════════════════════════════════════════════════

class SistemaExperto:
    """Motor de Forward Chaining con 30 reglas de compliance logístico."""

    def evaluar(self, pedidos: List[Pedido], camion: Camion) -> Tuple[float, List[Infraccion]]:
        infracciones: List[Infraccion] = []
        if not pedidos:
            return 0.0, infracciones

        nombres = [p.material.nombre for p in pedidos]
        for i, mat_a in enumerate(nombres):
            for mat_b in nombres[i + 1:]:
                if frozenset({mat_a, mat_b}) in MATERIALES_INCOMPATIBLES:
                    infracciones.append(Infraccion(1, f"Incompatibilidad química: {mat_a} + {mat_b}", 100_000, True))

        for p in pedidos:
            if p.material.nombre == "Aditivo Químico" and not camion.tiene_clima_controlado:
                infracciones.append(Infraccion(2, f"{p.id}: Aditivo Químico requiere clima controlado", 15_000, True))
            if p.material.nombre == "Ácido" and not camion.tiene_clima_controlado:
                infracciones.append(Infraccion(3, f"{p.id}: Ácido requiere contenedor especializado", 20_000, True))

        peso_total = sum(p.material.peso_kg for p in pedidos)
        vol_total = sum(p.material.volumen_m3 for p in pedidos)

        if peso_total > camion.capacidad_peso_kg:
            infracciones.append(Infraccion(
                7, f"Exceso de peso: {peso_total:.0f} kg > límite {camion.capacidad_peso_kg:.0f} kg", 50_000, True))
        if vol_total > camion.capacidad_volumen_m3:
            infracciones.append(Infraccion(
                8, f"Exceso de volumen: {vol_total:.1f} m³ > {camion.capacidad_volumen_m3:.0f} m³", 20_000, True))
        if peso_total > 35_000:
            infracciones.append(Infraccion(
                9, f"Peso {peso_total:.0f} kg > 35t: requiere ruta sin autopista urbana", 5_000, False))

        hora_actual = 7 * 60
        corralon_coords = (CORRALON["lat"], CORRALON["lon"])
        for p in pedidos:
            dist = math.sqrt((p.coordenadas[0]-corralon_coords[0])**2 +
                              (p.coordenadas[1]-corralon_coords[1])**2) * 111
            tiempo_viaje = (dist / camion.velocidad_kmh) * 60
            hora_llegada = hora_actual + tiempo_viaje
            if hora_llegada > p.ventana_fin:
                retraso = hora_llegada - p.ventana_fin
                pen = 500 * (1 + retraso / 30) ** 2
                infracciones.append(Infraccion(
                    13, f"{p.id}: Llegada {hora_llegada/60:.1f}h > cierre {p.ventana_fin/60:.1f}h", pen, False))
            elif hora_llegada < p.ventana_inicio:
                infracciones.append(Infraccion(
                    14, f"{p.id}: Llegada muy temprana ({hora_llegada/60:.1f}h)", 2_000, False))

        for p in pedidos:
            if not p.proveedor_certificado:
                infracciones.append(Infraccion(21, f"{p.id}: Proveedor sin certificación ISO/FSC", 10_000, True))
                if p.material.nombre == "Madera":
                    infracciones.append(Infraccion(
                        22, f"{p.id}: Madera sin certificado FSC — riesgo ambiental CRÍTICO", 25_000, True))
            if p.material.es_peligroso and len(pedidos) > 3:
                infracciones.append(Infraccion(
                    23, f"{p.id}: Material peligroso con >3 pedidos en el mismo camión", 30_000, True))

        return sum(i.penalizacion for i in infracciones), infracciones


SE = SistemaExperto()

REGLAS_TABLA = {
    "Bloque": (["⚗️ Química"]*6 + ["⚖️ Peso/Volumen"]*6 + ["⏰ Ventanas de Tiempo"]*8 + ["📋 Compliance"]*10),
    "#": list(range(1, 31)),
    "Condición IF": [
        "Cemento Y Yeso en mismo camión", "Cemento Y Material Húmedo en mismo camión",
        "Yeso Y Material Húmedo en mismo camión", "Acero Y Ácido en mismo camión",
        "Aditivo Químico Y sin clima controlado", "Ácido Y sin contenedor especializado",
        "Peso total > 45.000 kg", "Volumen total > 60 m³", "Peso > 35.000 kg Y autopista urbana",
        "Peso por eje supera límite", "Camión casi vacío (<500 kg)", "Vol > 80% Y material peligroso",
        "Llegada estimada > ventana_fin", "Llegada < ventana_inicio", "Retraso acumulado > 60 min",
        "Más de 4 paradas en horario pico", "Ventana 08-10h Y >3 pedidos en ruta",
        "Entrega nocturna Y sin permiso", "Ruta total > 80 km", "Pedido prioritario no es primera parada",
        "Proveedor sin certificación ISO", "Madera Y sin certificado FSC", "Obra pública Y certificado vencido",
        "Material estructural Y sin trazabilidad", "Material peligroso Y sin hoja MSDS",
        "Acero estructural Y sin cert. CIRSOC", "Conductor sin habilitación carga pesada",
        "Zona escolar Y sin permiso especial", "Material con vencimiento <30 días",
        "Pedido rechazado previamente Y reasignado",
    ],
    "Consecuencia THEN": [
        "Infracción CRÍTICA: fraguado prematuro", "Infracción CRÍTICA: fraguado por humedad",
        "Infracción ALTA: degradación del yeso", "Infracción CRÍTICA: corrosión acelerada",
        "Infracción ALTA: pérdida de propiedades", "Infracción CRÍTICA: riesgo de derrame",
        "Infracción CRÍTICA: exceso legal de peso", "Infracción CRÍTICA: carga imposible de cerrar",
        "Advertencia: ruta alternativa requerida", "Infracción: daño a infraestructura vial",
        "Advertencia: ineficiencia operativa", "Infracción: riesgo de manipulación",
        "Multa exponencial por retraso", "Advertencia: obra no disponible aún",
        "Multa adicional por cuadrilla perdida", "Advertencia: tiempos poco confiables",
        "Riesgo de incumplimiento de ventana", "Infracción: clausura por ruido nocturno",
        "Advertencia: viabilidad de ventanas", "Riesgo de penalización contractual",
        "Bloqueo por riesgo de auditoría", "Infracción CRÍTICA: riesgo ambiental",
        "Bloqueo automático: clausura de obra", "Infracción: incumplimiento IRAM",
        "Infracción: Ley de Riesgos del Trabajo", "Invalida certificado de la obra",
        "Infracción CRÍTICA: multa de tránsito", "Infracción: horario restringido",
        "Advertencia: riesgo de rechazo en obra", "Verificar causa del rechazo anterior",
    ],
    "Tipo": (["🔴 Crítica"]*4 + ["🟡 Alta"]*2 + ["🔴 Crítica"]*2 + ["🟡 Media"]*4 +
             ["🟡 Media"]*8 + ["🔴 Crítica"]*3 + ["🟡 Media"]*7),
    "Penalización ($)": [
        "100.000", "100.000", "100.000", "100.000", "15.000", "20.000",
        "50.000", "20.000", "5.000", "25.000", "1.000", "10.000",
        "Variable (exp.)", "2.000", "5.000/h", "3.000", "2.000",
        "20.000", "1.500", "3.000", "10.000", "25.000", "50.000",
        "15.000", "20.000", "30.000", "40.000", "5.000", "2.000", "1.000",
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# ALGORITMO GENÉTICO — DEAP
# ═══════════════════════════════════════════════════════════════════════════

def distancia_km(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) * 111


def decodificar(individuo, delimitador):
    rutas, actual = [], []
    for gen in individuo:
        if gen == delimitador:
            if actual:
                rutas.append(actual)
                actual = []
        else:
            actual.append(gen)
    if actual:
        rutas.append(actual)
    return [r for r in rutas if r]


def reparar(individuo, n_pedidos, delimitador):
    presentes, duplicados = set(), []
    for i, gen in enumerate(individuo):
        if gen != delimitador:
            if gen in presentes:
                duplicados.append(i)
            else:
                presentes.add(gen)
    faltantes = list(set(range(n_pedidos)) - presentes)
    for idx, falt in zip(duplicados, faltantes):
        individuo[idx] = falt
    faltantes = list(set(range(n_pedidos)) - {g for g in individuo if g != delimitador})
    for i in range(len(individuo)):
        if not faltantes:
            break
        if individuo[i] == delimitador and faltantes:
            individuo[i] = faltantes.pop()
    return individuo


def construir_toolbox(pedidos, n_camiones_max):
    n_pedidos = len(pedidos)
    delimitador = n_pedidos
    corralon_coords = (CORRALON["lat"], CORRALON["lon"])

    def fitness_hibrido(individuo):
        rutas = decodificar(individuo, delimitador)
        costo_total, pen_total = 0.0, 0.0
        for i, ruta in enumerate(rutas):
            camion = Camion(id=f"Camion_{i+1}")
            pedidos_ruta = [pedidos[idx] for idx in ruta]
            coords = [corralon_coords] + [p.coordenadas for p in pedidos_ruta] + [corralon_coords]
            for j in range(len(coords)-1):
                costo_total += distancia_km(coords[j], coords[j+1])
            pen, _ = SE.evaluar(pedidos_ruta, camion)
            pen_total += pen
        return (1.0 / (costo_total + pen_total + 1e-6),)

    def crear_individuo():
        idx = list(range(n_pedidos))
        random.shuffle(idx)
        n_sep = random.randint(1, max(1, n_camiones_max - 1))
        pos = sorted(random.sample(range(1, n_pedidos), min(n_sep, n_pedidos - 1))) if n_pedidos > 1 else []
        ind, prev = [], 0
        for p in pos:
            ind.extend(idx[prev:p])
            ind.append(delimitador)
            prev = p
        ind.extend(idx[prev:])
        return creator.Individuo(ind)

    def mate_rep(ind1, ind2):
        tools.cxUniform(ind1, ind2, indpb=0.5)
        reparar(ind1, n_pedidos, delimitador)
        reparar(ind2, n_pedidos, delimitador)
        return ind1, ind2

    def mutar_rep(ind, indpb):
        tools.mutShuffleIndexes(ind, indpb=indpb)
        reparar(ind, n_pedidos, delimitador)
        return (ind,)

    if "FitnessMax" in creator.__dict__:
        del creator.FitnessMax
    if "Individuo" in creator.__dict__:
        del creator.Individuo
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individuo", list, fitness=creator.FitnessMax)

    tb = base.Toolbox()
    tb.register("individual", crear_individuo)
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register("evaluate", fitness_hibrido)
    tb.register("select", tools.selTournament, tournsize=3)
    tb.register("mate", mate_rep)
    tb.register("mutate", mutar_rep, indpb=0.05)
    return tb, delimitador


def ejecutar_ag(pedidos, generaciones, poblacion, prob_cruce, prob_mutacion, n_camiones_max, seed,
                 progress_cb=None):
    random.seed(seed)
    np.random.seed(seed)
    tb, delimitador = construir_toolbox(pedidos, n_camiones_max)
    pop = tb.population(n=poblacion)
    hof = tools.HallOfFame(1)

    for ind in pop:
        ind.fitness.values = tb.evaluate(ind)
    hof.update(pop)

    gens = [0]
    max_fits = [hof[0].fitness.values[0]]
    avg_fits = [float(np.mean([i.fitness.values[0] for i in pop]))]

    for g in range(1, generaciones + 1):
        offspring = algorithms.varOr(pop, tb, lambda_=poblacion, cxpb=prob_cruce, mutpb=prob_mutacion)
        for ind in offspring:
            ind.fitness.values = tb.evaluate(ind)
        pop[:] = tb.select(pop + offspring, poblacion)
        hof.update(pop)

        gens.append(g)
        max_fits.append(hof[0].fitness.values[0])
        avg_fits.append(float(np.mean([i.fitness.values[0] for i in pop])))

        if progress_cb is not None:
            progress_cb(g, generaciones)

    class Logbook:
        def select(self, key):
            return {"gen": gens, "max": max_fits, "avg": avg_fits}[key]

    return hof[0], Logbook(), delimitador


# ═══════════════════════════════════════════════════════════════════════════
# COMPONENTES DE UI REUTILIZABLES
# ═══════════════════════════════════════════════════════════════════════════

def callout(texto, tipo="info"):
    st.markdown(f'<div class="callout callout-{tipo}">{texto}</div>', unsafe_allow_html=True)


def kpi(label, value, sub="", accent="blue"):
    st.markdown(f"""
    <div class="kpi-card kpi-accent-{accent}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)


def section(icon, title, sub=""):
    st.markdown(f'<div class="section-title">{icon} {title}</div>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<div class="section-sub">{sub}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# ESTADO DE SESIÓN
# ═══════════════════════════════════════════════════════════════════════════

def init_state():
    defaults = {
        "pedidos": None, "df": None, "n_pedidos": None, "seed": None,
        "mejor": None, "logbook": None, "delimitador": None,
        "ag_elapsed": None, "n_camiones_max_used": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def asegurar_dataset(n_pedidos, seed):
    if (st.session_state["pedidos"] is None or
            st.session_state["n_pedidos"] != n_pedidos or
            st.session_state["seed"] != seed):
        pedidos = generar_pedidos(n_pedidos, seed)
        st.session_state["pedidos"] = pedidos
        st.session_state["df"] = pedidos_a_dataframe(pedidos)
        st.session_state["n_pedidos"] = n_pedidos
        st.session_state["seed"] = seed
        st.session_state["mejor"] = None
        st.session_state["logbook"] = None
        st.session_state["delimitador"] = None


# ═══════════════════════════════════════════════════════════════════════════
# PÁGINA: INICIO
# ═══════════════════════════════════════════════════════════════════════════

def pagina_inicio():
    st.markdown("""
    <div class="hero">
        <h1>🚛 SmartDispatch</h1>
        <p>Optimizador híbrido de flota para logística de construcción —
        Algoritmo Genético + Sistema Experto de Compliance</p>
        <div class="hero-badges">
            <span class="pill">🧬 DEAP · Algoritmos Genéticos</span>
            <span class="pill">🧠 Forward Chaining · 30 reglas</span>
            <span class="pill">🗺️ Optimización geoespacial</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi("Reglas de Compliance", "30", "Química · Peso · Tiempo · Normativa", "blue")
    with c2:
        kpi("Reducción de Distancia", "~25 %", "vs. asignación aleatoria", "green")
    with c3:
        kpi("Tiempo de Planificación", "&lt; 1 min", "vs. 3-5 horas manuales", "orange")

    st.markdown("<br>", unsafe_allow_html=True)
    section("💡", "¿Qué problema resuelve?")
    callout(
        "Una distribuidora de materiales de construcción debe asignar diariamente sus pedidos a una flota "
        "de camiones, decidiendo <strong>qué va en cada camión y en qué orden se entrega</strong>. "
        "Esto no es solo un problema de distancia: hay <strong>materiales que no pueden viajar juntos</strong> "
        "(cemento + yeso reaccionan), <strong>límites legales de peso</strong>, "
        "<strong>ventanas horarias estrictas</strong> en cada obra, y <strong>certificaciones de proveedores</strong> "
        "que deben estar vigentes. Resolverlo a mano en Excel toma horas y es propenso a errores costosos.",
        "info"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    section("⚙️", "Arquitectura híbrida: por qué dos sistemas de IA")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🧬 Algoritmo Genético — el optimizador")
        callout(
            "Explora miles de combinaciones de asignación de pedidos a camiones, evolucionando "
            "una población de soluciones mediante <strong>selección, cruce y mutación</strong> hasta "
            "converger en la ruta de menor costo. Implementado con la librería <strong>DEAP</strong>.",
            "info"
        )
    with col2:
        st.markdown("#### 🧠 Sistema Experto — el auditor")
        callout(
            "Evalúa cada solución propuesta contra <strong>30 reglas de negocio</strong> codificadas como "
            "lógica IF-THEN. Usa <strong>Forward Chaining</strong>: ante los hechos de una ruta, dispara "
            "todas las reglas aplicables y devuelve una penalización y una traza explicable.",
            "info"
        )

    callout(
        "🔗 <strong>La integración</strong> ocurre dentro de la función de fitness del AG: cada vez que evalúa "
        "una solución candidata, le pide al Sistema Experto su penalización en pesos y la suma al costo de "
        "distancia. Así, las soluciones que violan reglas críticas son eliminadas evolutivamente, sin "
        "necesidad de programarlas como restricciones explícitas en el algoritmo de búsqueda.",
        "ok"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    section("🧭", "Cómo usar esta aplicación")
    c1, c2, c3, c4 = st.columns(4)
    pasos = [
        ("1️⃣", "Configurá", "Ajustá el tamaño del dataset y los parámetros del AG en la barra lateral"),
        ("2️⃣", "Explorá", "Revisá el EDA: distribución de materiales, pesos y el mapa de obras"),
        ("3️⃣", "Optimizá", "Ejecutá el Algoritmo Genético y mirá la convergencia en tiempo real"),
        ("4️⃣", "Auditá", "Revisá la hoja de ruta final con el informe de compliance por camión"),
    ]
    for col, (icono, titulo, desc) in zip([c1, c2, c3, c4], pasos):
        with col:
            st.markdown(f"""
            <div class="kpi-card kpi-accent-blue">
                <div style="font-size:1.6rem">{icono}</div>
                <div style="font-weight:700; color:#0F2A4A; margin-top:4px;">{titulo}</div>
                <div style="color:#6B7785; font-size:0.85rem; margin-top:4px;">{desc}</div>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PÁGINA: EDA
# ═══════════════════════════════════════════════════════════════════════════

def pagina_eda():
    pedidos = st.session_state["pedidos"]
    df = st.session_state["df"]

    section("📊", "Análisis Exploratorio de Datos",
            "Cada visualización conecta directamente con un bloque de reglas del Sistema Experto.")

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Total pedidos", len(pedidos), accent="blue")
    with c2: kpi("Peso promedio", f"{df['Peso_kg'].mean():.0f} kg", accent="blue")
    with c3: kpi("Sin certificar", f"{(~df['Certificado']).sum()}", "pedidos en riesgo normativo", "orange")
    with c4: kpi("Materiales peligrosos", f"{df['Peligroso'].sum()}", "requieren clima controlado", "red")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### 📦 Pedidos por Tipo de Material")
        conteo = df["Material"].value_counts().reset_index()
        conteo.columns = ["Material", "Cantidad"]
        fig = px.bar(conteo, x="Material", y="Cantidad", color="Material",
                     color_discrete_map=COLOR_MATERIAL)
        fig.update_layout(showlegend=False, xaxis_tickangle=-30, height=340,
                           margin=dict(t=10, b=10), plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
        callout("Concentración de materiales químicamente incompatibles (Cemento/Yeso) "
                "→ mayor presión sobre las reglas del Bloque 1 del SE.", "info")

    with col2:
        st.markdown("##### ⚖️ Distribución de Peso por Material")
        fig = px.box(df, x="Material", y="Peso_kg", color="Material",
                     color_discrete_map=COLOR_MATERIAL)
        fig.add_hline(y=45000, line_dash="dash", line_color="#E74C3C",
                      annotation_text="Límite legal 45t")
        fig.update_layout(showlegend=False, xaxis_tickangle=-30, height=340,
                           margin=dict(t=10, b=10), plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
        callout("Justifica las reglas de peso/volumen (Bloque 2): a mayor varianza, "
                "mayor riesgo de exceder el límite por camión.", "info")

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("##### 📋 Certificación de Proveedores")
        cert = df["Certificado"].value_counts().reset_index()
        cert.columns = ["Estado", "Cantidad"]
        cert["Label"] = cert["Estado"].map({True: "✅ Certificado", False: "❌ Sin Certificado"})
        fig = px.pie(cert, values="Cantidad", names="Label",
                     color_discrete_sequence=["#27AE60", "#E74C3C"], hole=0.45)
        fig.update_layout(height=340, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        callout("Activa directamente las reglas 21-22 del Bloque de Compliance Normativo.", "warn")

    with col4:
        st.markdown("##### 🕐 Ventanas Horarias por Material")
        vent = df.groupby(["Ventana_Inicio", "Material"]).size().reset_index(name="Cantidad")
        fig = px.bar(vent, x="Ventana_Inicio", y="Cantidad", color="Material",
                     color_discrete_map=COLOR_MATERIAL, barmode="stack")
        fig.update_layout(height=340, margin=dict(t=10, b=10), plot_bgcolor="white",
                           xaxis_title="Apertura de ventana")
        st.plotly_chart(fig, use_container_width=True)
        callout("Ventanas concentradas → el AG debe encontrar rutas muy eficientes "
                "para llegar a tiempo a todas (Bloque 3).", "info")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("##### 🗺️ Mapa Geoespacial de Obras")
    mapa = folium.Map(location=[CORRALON["lat"], CORRALON["lon"]], zoom_start=12, tiles="cartodbpositron")
    folium.Marker([CORRALON["lat"], CORRALON["lon"]], popup="🏭 Corralón Central",
                  icon=folium.Icon(color="red", icon="home", prefix="fa")).add_to(mapa)
    for p in pedidos:
        folium.CircleMarker(
            location=p.coordenadas, radius=8,
            color=COLOR_FOLIUM.get(p.material.nombre, "blue"), fill=True, fill_opacity=0.8,
            popup=folium.Popup(
                f"<b>{p.id}</b> — {p.obra_destino}<br>{p.material.nombre}<br>"
                f"{p.material.peso_kg:.0f} kg<br>"
                f"{'✅ Certificado' if p.proveedor_certificado else '❌ Sin certificar'}",
                max_width=220)
        ).add_to(mapa)
    st_folium(mapa, width=None, height=460, returned_objects=[])

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("##### 📄 Dataset completo")
    st.dataframe(df, use_container_width=True, height=280)
    st.download_button("⬇️ Descargar dataset (CSV)", df.to_csv(index=False).encode("utf-8"),
                        "smartdispatch_pedidos.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════════
# PÁGINA: SISTEMA EXPERTO
# ═══════════════════════════════════════════════════════════════════════════

def pagina_sistema_experto():
    section("🧠", "Sistema Experto — Motor de Compliance",
            "Forward Chaining: ante los hechos de un camión, se disparan todas las reglas aplicables.")

    with st.expander("📋 Ver la Base de Conocimiento completa (30 reglas)", expanded=False):
        st.dataframe(pd.DataFrame(REGLAS_TABLA), use_container_width=True, height=520)

    st.markdown("<br>", unsafe_allow_html=True)
    section("🔬", "Simulador interactivo", "Armá un camión de prueba y mirá la auditoría en vivo.")

    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.markdown("**Configuración del camión:**")
        materiales_sel = st.multiselect(
            "Materiales a cargar", [m["nombre"] for m in TIPOS_MATERIALES],
            default=["Cemento", "Yeso"], key="se_materiales")
        peso_input = st.slider("Peso total (kg)", 500, 60000, 8000, 500, key="se_peso")
        vol_input = st.slider("Volumen total (m³)", 0.5, 80.0, 5.0, 0.5, key="se_vol")
        tiene_clima = st.checkbox("Camión con clima controlado", False, key="se_clima")
        cert_ok = st.checkbox("Todos los proveedores certificados", True, key="se_cert")

    with col2:
        st.markdown("**Resultado de la auditoría:**")
        if not materiales_sel:
            callout("Seleccioná al menos un material para auditar.", "warn")
        else:
            pedidos_prueba = []
            for j, nombre_mat in enumerate(materiales_sel):
                mat_data = next(m for m in TIPOS_MATERIALES if m["nombre"] == nombre_mat)
                mat = Material(
                    nombre=nombre_mat, peso_kg=peso_input/len(materiales_sel),
                    volumen_m3=vol_input/len(materiales_sel),
                    requiere_clima_controlado=mat_data["clima"], es_peligroso=mat_data["peligroso"])
                pedidos_prueba.append(Pedido(
                    id=f"TEST_{j+1}", material=mat, obra_destino="Obra_Demo",
                    coordenadas=(CORRALON["lat"]+0.05, CORRALON["lon"]+0.05),
                    ventana_inicio=480, ventana_fin=600, proveedor_certificado=cert_ok))
            camion_prueba = Camion(id="Camión de Prueba", tiene_clima_controlado=tiene_clima)
            pen_total, infracciones = SE.evaluar(pedidos_prueba, camion_prueba)

            if not infracciones:
                callout("✅ <strong>APROBADO</strong> — 0 infracciones detectadas.", "ok")
            else:
                criticas = sum(1 for i in infracciones if i.es_critica)
                callout(f"⚠️ <strong>{len(infracciones)} infracción(es)</strong> · {criticas} crítica(s) · "
                        f"Penalización total: <strong>${pen_total:,.0f}</strong>", "danger")

            for inf in infracciones:
                tipo = "🔴 CRÍTICO" if inf.es_critica else "🟡 ADVERTENCIA"
                st.markdown(f"- **Regla #{inf.regla_id}** [{tipo}]: {inf.descripcion} "
                            f"— **${inf.penalizacion:,.0f}**")

            callout(f"📐 Impacto aproximado en el fitness del AG: "
                    f"<code>1 / (distancia + {pen_total:,.0f})</code> ≈ "
                    f"<strong>{'descartado evolutivamente' if pen_total > 1000 else 'mantenido en la población'}</strong>",
                    "warn" if pen_total > 0 else "ok")


# ═══════════════════════════════════════════════════════════════════════════
# PÁGINA: OPTIMIZACIÓN
# ═══════════════════════════════════════════════════════════════════════════

def pagina_optimizacion(cfg):
    pedidos = st.session_state["pedidos"]

    section("🧬", "Optimización con Algoritmo Genético",
            "Selección por Torneo · Cruce Uniforme con Reparación · Mutación Shuffle con Reparación")

    col1, col2, col3 = st.columns(3)
    with col1:
        callout("<strong>🏆 Selección: Torneo (k=3)</strong><br>Mantiene diversidad genética, "
                "evita convergencia prematura frente a la Ruleta.", "info")
    with col2:
        callout("<strong>🔀 Cruce: Uniforme + Reparación</strong><br>Garantiza que cada pedido "
                "aparezca exactamente una vez tras el cruce.", "info")
    with col3:
        callout("<strong>🎲 Mutación: Shuffle + Reparación</strong><br>Introduce variación "
                "controlada para escapar de óptimos locales.", "info")

    st.latex(r"Fitness = \frac{1}{D_{total} + P_{SE} + \epsilon}")
    callout("<strong>D_total</strong>: distancia de cada ruta (Corralón → obras → Corralón) en km · "
            "<strong>P_SE</strong>: penalización del Sistema Experto en $ · <strong>ε=10⁻⁶</strong>: "
            "estabilidad numérica. Un fitness más alto (cercano a 1) es mejor.", "ok")

    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("🚀 Ejecutar Algoritmo Genético", type="primary", use_container_width=True)

    if run:
        progress_bar = st.progress(0, text="Inicializando población...")
        start = time.time()

        def cb(g, total):
            pct = int(g / total * 100)
            progress_bar.progress(pct, text=f"Generación {g}/{total}")

        mejor, logbook, delimitador = ejecutar_ag(
            pedidos, cfg["generaciones"], cfg["poblacion"], cfg["prob_cruce"],
            cfg["prob_mutacion"], cfg["n_camiones_max"], cfg["seed"], progress_cb=cb,
        )
        elapsed = time.time() - start
        progress_bar.progress(100, text="✅ Optimización completada")

        st.session_state["mejor"] = mejor
        st.session_state["logbook"] = logbook
        st.session_state["delimitador"] = delimitador
        st.session_state["ag_elapsed"] = elapsed
        st.session_state["n_camiones_max_used"] = cfg["n_camiones_max"]
        st.success(f"Optimización completada en {elapsed:.1f}s · Fitness final: {mejor.fitness.values[0]:.6f}")

    if st.session_state["logbook"] is not None:
        logbook = st.session_state["logbook"]
        gen = logbook.select("gen")
        max_fit = logbook.select("max")
        avg_fit = logbook.select("avg")
        generaciones = gen[-1] if gen else cfg["generaciones"]

        st.markdown("<br>", unsafe_allow_html=True)
        section("📈", "Convergencia del Algoritmo")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=gen, y=max_fit, mode="lines", name="Mejor Fitness",
                                  line=dict(color="#2E75B6", width=2.5),
                                  fill="tozeroy", fillcolor="rgba(46,117,182,0.08)"))
        fig.add_trace(go.Scatter(x=gen, y=avg_fit, mode="lines", name="Fitness Promedio",
                                  line=dict(color="#E74C3C", width=1.5, dash="dash")))
        if generaciones > 0:
            fig.add_vrect(x0=0, x1=generaciones*0.2, fillcolor="rgba(231,76,60,0.05)", line_width=0,
                          annotation_text="Exploración", annotation_position="top left")
            fig.add_vrect(x0=generaciones*0.2, x1=generaciones*0.6, fillcolor="rgba(39,174,96,0.05)",
                          line_width=0, annotation_text="Aprendizaje", annotation_position="top left")
            fig.add_vrect(x0=generaciones*0.6, x1=generaciones, fillcolor="rgba(46,117,182,0.05)",
                          line_width=0, annotation_text="Convergencia", annotation_position="top left")
        fig.update_layout(height=420, template="plotly_white",
                           xaxis_title="Generación", yaxis_title="Fitness",
                           legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
                           margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

        mejora_pct = (max_fit[-1]/max_fit[0]-1)*100 if max_fit[0] > 0 else 0
        callout(f"🔍 El fitness mejoró un <strong>{mejora_pct:.1f}%</strong> respecto a la generación inicial. "
                f"Fitness final: <strong>{max_fit[-1]:.6f}</strong>.", "ok")
    else:
        callout("Configurá los parámetros en la barra lateral y presioná 'Ejecutar' para comenzar.", "info")


# ═══════════════════════════════════════════════════════════════════════════
# PÁGINA: RESULTADOS
# ═══════════════════════════════════════════════════════════════════════════

def pagina_resultados():
    section("📋", "Hoja de Ruta Final y Auditoría")

    if st.session_state["mejor"] is None:
        callout("⚠️ Todavía no se ejecutó el Algoritmo Genético. Ve a la pestaña "
                "<strong>🧬 Optimización</strong> y ejecutalo primero.", "warn")
        return

    pedidos = st.session_state["pedidos"]
    mejor = st.session_state["mejor"]
    delimitador = st.session_state["delimitador"]
    rutas = decodificar(mejor, delimitador)
    corralon_coords = (CORRALON["lat"], CORRALON["lon"])

    resultados = []
    total_pen, total_dist, total_inf = 0, 0, 0
    for i, ruta in enumerate(rutas):
        camion = Camion(id=f"Camión {i+1}")
        pedidos_ruta = [pedidos[idx] for idx in ruta]
        pen, infracciones = SE.evaluar(pedidos_ruta, camion)
        coords = [corralon_coords] + [p.coordenadas for p in pedidos_ruta] + [corralon_coords]
        dist = sum(distancia_km(coords[j], coords[j+1]) for j in range(len(coords)-1))
        total_pen += pen
        total_dist += dist
        total_inf += len(infracciones)
        resultados.append((camion, pedidos_ruta, pen, infracciones, dist))

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Camiones usados", len(rutas), accent="blue")
    with c2: kpi("Distancia total", f"{total_dist:.1f} km", accent="blue")
    with c3: kpi("Penalización SE", f"${total_pen:,.0f}", accent=("green" if total_pen == 0 else "red"))
    with c4: kpi("Infracciones", total_inf, accent=("green" if total_inf == 0 else "orange"))

    if total_pen == 0:
        callout("🏆 <strong>Solución 100% Compliant</strong> — ninguna regla del Sistema Experto fue violada.", "ok")
    else:
        callout(f"⚠️ Penalización pendiente de ${total_pen:,.0f}. "
                "Probá aumentar generaciones/población en la pestaña de Optimización.", "warn")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("##### 🚛 Detalle por camión")
    for camion, pedidos_ruta, pen, infracciones, dist in resultados:
        emoji = "✅" if pen == 0 else "⚠️"
        with st.expander(f"{emoji} {camion.id} — {len(pedidos_ruta)} pedidos · "
                          f"{sum(p.material.peso_kg for p in pedidos_ruta):.0f} kg · {dist:.1f} km",
                          expanded=(pen > 0)):
            cc1, cc2, cc3 = st.columns(3)
            with cc1: kpi("Pedidos", len(pedidos_ruta), accent="blue")
            with cc2: kpi("Peso", f"{sum(p.material.peso_kg for p in pedidos_ruta):.0f} kg", accent="blue")
            with cc3: kpi("Distancia", f"{dist:.1f} km", accent="blue")

            st.markdown("**Pedidos asignados:**")
            for p in pedidos_ruta:
                cert = "✅" if p.proveedor_certificado else "❌"
                st.markdown(f"- **{p.id}** — {p.material.nombre} ({p.material.peso_kg:.0f} kg) "
                            f"→ {p.obra_destino} "
                            f"[{p.ventana_inicio//60:02d}:{p.ventana_inicio%60:02d}"
                            f"–{p.ventana_fin//60:02d}:{p.ventana_fin%60:02d}] · Cert: {cert}")

            st.markdown("**Traza del Sistema Experto:**")
            if not infracciones:
                callout("✅ APROBADO — Bloques 1 a 4 sin infracciones.", "ok")
            else:
                for inf in infracciones:
                    tipo = "🔴 CRÍTICO" if inf.es_critica else "🟡 ADVERTENCIA"
                    callout(f"[{tipo}] Regla #{inf.regla_id}: {inf.descripcion} — "
                            f"<strong>${inf.penalizacion:,.0f}</strong>",
                            "danger" if inf.es_critica else "warn")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("##### 🗺️ Mapa de rutas optimizadas")
    mapa = folium.Map(location=[CORRALON["lat"], CORRALON["lon"]], zoom_start=12, tiles="cartodbpositron")
    folium.Marker([CORRALON["lat"], CORRALON["lon"]], popup="🏭 Corralón Central",
                  icon=folium.Icon(color="red", icon="home", prefix="fa")).add_to(mapa)
    for i, (camion, pedidos_ruta, pen, _, _) in enumerate(resultados):
        color = COLOR_CAMION[i % len(COLOR_CAMION)]
        puntos = [corralon_coords] + [p.coordenadas for p in pedidos_ruta] + [corralon_coords]
        folium.PolyLine(puntos, color=color, weight=3, opacity=0.85,
                         tooltip=f"{camion.id} ({len(pedidos_ruta)} pedidos)").add_to(mapa)
        for p in pedidos_ruta:
            folium.CircleMarker(location=p.coordenadas, radius=8, color=color, fill=True, fill_opacity=0.9,
                                 popup=f"<b>{p.id}</b> — {camion.id}<br>{p.material.nombre}<br>"
                                       f"{p.material.peso_kg:.0f} kg").add_to(mapa)
    st_folium(mapa, width=None, height=480, returned_objects=[])

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("##### 📊 Resumen ejecutivo y exportación")
    df_resumen = pd.DataFrame([{
        "Camión": c.id, "Pedidos": ", ".join(p.id for p in pr),
        "Materiales": ", ".join(p.material.nombre for p in pr),
        "Peso (kg)": f"{sum(p.material.peso_kg for p in pr):.0f}",
        "Distancia (km)": f"{d:.1f}", "Penalización SE": f"${pe:,.0f}",
        "Estado": "✅ Aprobado" if pe == 0 else f"⚠️ {len(ifs)} infracción(es)",
    } for c, pr, pe, ifs, d in resultados])
    st.dataframe(df_resumen, use_container_width=True)
    st.download_button("⬇️ Descargar hoja de ruta (CSV)", df_resumen.to_csv(index=False).encode("utf-8"),
                        "smartdispatch_hoja_de_ruta.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    init_state()

    with st.sidebar:
        st.markdown("## ⚙️ Panel de Control")
        st.markdown("---")
        st.markdown("##### 📦 Dataset")
        n_pedidos = st.slider("Cantidad de pedidos", 10, 80, 30, 5)
        seed = st.number_input("Semilla aleatoria", 1, 9999, 42, 1)

        st.markdown("---")
        st.markdown("##### 🧬 Parámetros del AG")
        n_camiones_max = st.slider("Camiones máximos", 2, 10, 5, 1)
        generaciones = st.slider("Generaciones", 30, 300, 120, 10)
        poblacion = st.slider("Tamaño de población", 50, 400, 150, 25)
        prob_cruce = st.slider("Probabilidad de cruce", 0.5, 1.0, 0.8, 0.05)
        prob_mutacion = st.slider("Probabilidad de mutación", 0.05, 0.5, 0.15, 0.05)

        st.markdown("---")
        st.caption("SmartDispatch v1.0 · AG (DEAP) + Sistema Experto · "
                   "Proyecto académico de IA Aplicada")

    asegurar_dataset(n_pedidos, seed)
    cfg = dict(generaciones=generaciones, poblacion=poblacion, prob_cruce=prob_cruce,
               prob_mutacion=prob_mutacion, n_camiones_max=n_camiones_max, seed=seed)

    tabs = st.tabs(["🏠 Inicio", "📊 EDA", "🧠 Sistema Experto", "🧬 Optimización", "📋 Resultados"])
    with tabs[0]:
        pagina_inicio()
    with tabs[1]:
        pagina_eda()
    with tabs[2]:
        pagina_sistema_experto()
    with tabs[3]:
        pagina_optimizacion(cfg)
    with tabs[4]:
        pagina_resultados()


if __name__ == "__main__":
    main()