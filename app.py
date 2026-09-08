import json
from google import genai
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE LA PÁGINA Y ESTILOS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Chancay - Analítica Prescriptiva",
    page_icon="🏗️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-header {font-size:2.2rem; font-weight:bold; color:#1E3A8A; margin-bottom:0px;}
    .sub-header {font-size:1.1rem; color:#4B5563; margin-bottom:20px;}
    .metric-card {background-color:#F3F4F6; padding:15px; border-radius:10px; text-align:center;}
    </style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 2. CARGA Y PROCESAMIENTO DE DATOS (MÓDULO 6 Y 7)
# -----------------------------------------------------------------------------


@st.cache_data
def cargar_datos_operarios():
  np.random.seed(42)
  n = 100

  avance_pct = np.random.uniform(40, 105, n).round(1)
  dias_entrenamiento = np.random.randint(10, 90, n)
  es_temporada_alta = np.random.choice([0, 1], size=n, p=[0.6, 0.4])
  evaluacion_desempeno = np.random.uniform(1.0, 5.0, n).round(2)

  # Cálculo probabilístico simplificado simulación XGBoost
  logits = (
      -0.04 * (avance_pct - 70)
      + 1.5 * es_temporada_alta
      - 0.8 * (evaluacion_desempeno - 3)
      + np.random.normal(0, 0.5, n)
  )
  prob = 1 / (1 + np.exp(-logits))
  prob_pct = (prob * 100).round(2)

  df = pd.DataFrame({
      "operario_id": [f"OP-{1000 + i}" for i in range(n)],
      "nombre": [f"Operario {i+1}" for i in range(n)],
      "avance_entrenamiento_pct": avance_pct,
      "dias_entrenamiento": dias_entrenamiento,
      "es_temporada_alta_agricola": es_temporada_alta,
      "evaluacion_desempeno": evaluacion_desempeno,
      "probabilidad_cese_pct": prob_pct,
  })

  # Asignación de Nivel de Riesgo
  conditions = [
      (df["probabilidad_cese_pct"] >= 70),
      (df["probabilidad_cese_pct"] >= 40)
      & (df["probabilidad_cese_pct"] < 70),
      (df["probabilidad_cese_pct"] < 40),
  ]
  choices = ["CRÍTICO", "MEDIO", "BAJO"]
  df["nivel_riesgo"] = np.select(conditions, choices, default="BAJO")

  return df


df = cargar_datos_operarios()

# -----------------------------------------------------------------------------
# 3. BARRA LATERAL: FILTROS DINÁMICOS
# -----------------------------------------------------------------------------
st.sidebar.header("🕹️ Panel de Control")

# Filtro por nivel de riesgo
riesgos_seleccionados = st.sidebar.multiselect(
    "Filtrar por Nivel de Riesgo:",
    options=["CRÍTICO", "MEDIO", "BAJO"],
    default=["CRÍTICO", "MEDIO"],
)

df_filtrado = df[df["nivel_riesgo"].isin(riesgos_seleccionados)]

if df_filtrado.empty:
  st.warning("No hay operarios que coincidan con los filtros seleccionados.")
  st.stop()

# Selección individual de operario
operario_id_sel = st.sidebar.selectbox(
    "Seleccionar Colaborador:", options=df_filtrado["operario_id"].tolist()
)

# Datos del operario seleccionado
trabajador = df_filtrado[df_filtrado["operario_id"] == operario_id_sel].iloc[0]

# -----------------------------------------------------------------------------
# 4. ENCABEZADO Y KPIS GENERALES
# -----------------------------------------------------------------------------
st.markdown(
    '<p class="main-header">🏗️ Dashboard Chancay: Retención de Talento</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="sub-header">Sistema de Analítica Prescriptiva impulsado por'
    " XGBoost, SHAP y Gemini 3.8 Flash</p>",
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Muestra", f"{len(df)} operarios")
col2.metric(
    "Casos Críticos (>70%)",
    f"{len(df[df['nivel_riesgo'] == 'CRÍTICO'])}",
    delta_color="inverse",
)
col3.metric(
    "Riesgo Promedio", f"{df['probabilidad_cese_pct'].mean():.1f}%"
)
col4.metric(
    "Expuestos a Temporada Alta",
    f"{len(df[df['es_temporada_alta_agricola'] == 1])}",
)

st.divider()

# -----------------------------------------------------------------------------
# 5. EXPEDIENTE INDIVIDUAL Y GRÁFICOS DINÁMICOS
# -----------------------------------------------------------------------------
st.subheader(
    f"👤 Expediente Operativo: {trabajador['nombre']} ({trabajador['operario_id']})"
)

col_gauge, col_metrics = st.columns([1, 2])

with col_gauge:
  # Gráfico de Tacómetro (Gauge) para Probabilidad de Cese
  color_gauge = (
      "#D9534F"
      if trabajador["nivel_riesgo"] == "CRÍTICO"
      else ("#F0AD4E" if trabajador["nivel_riesgo"] == "MEDIO" else "#5CB85C")
  )

  fig_gauge = go.Figure(
      go.Indicator(
          mode="gauge+number",
          value=trabajador["probabilidad_cese_pct"],
          title={"text": "Riesgo de Cese Estimado"},
          number={"suffix": "%"},
          gauge={
              "axis": {"range": [0, 100]},
              "bar": {"color": color_gauge},
              "steps": [
                  {"range": [0, 40], "color": "#E8F5E9"},
                  {"range": [40, 70], "color": "#FFF3E0"},
                  {"range": [70, 100], "color": "#FFEBEE"},
              ],
          },
      )
  )
  fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=20))
  st.plotly_chart(fig_gauge, use_container_width=True)

with col_metrics:
  st.markdown("##### Variables Operativas de Entrada")
  m1, m2, m3 = st.columns(3)
  m1.metric("Evaluación Desempeño", f"{trabajador['evaluacion_desempeno']} / 5.0")
  m2.metric("Avance Entrenamiento", f"{trabajador['avance_entrenamiento_pct']}%")
  m3.metric("Días en Planta", f"{trabajador['dias_entrenamiento']} días")

  # Gráfico de barras comparativo de perfil
  perfil_df = pd.DataFrame({
      "Métrica": [
          "Desempeño (x20)",
          "Avance Training (%)",
          "Días en Planta",
      ],
      "Valor": [
          trabajador["evaluacion_desempeno"] * 20,
          trabajador["avance_entrenamiento_pct"],
          trabajador["dias_entrenamiento"],
      ],
  })
  fig_bar = px.bar(
      perfil_df,
      x="Métrica",
      y="Valor",
      color="Métrica",
      title="Comparativo de Métricas Clave",
  )
  fig_bar.update_layout(
      height=200, showlegend=False, margin=dict(l=20, r=20, t=30, b=20)
  )
  st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# -----------------------------------------------------------------------------
# 6. MÓDULO PRESCRIPTIVO: GENERACIÓN CON GEMINI AI
# -----------------------------------------------------------------------------
st.subheader("🤖 Diagnóstico Prescriptivo e Intervención")
st.write(
    "Genera el plan de acción estructurado para el supervisor utilizando el"
    " LLM con contexto de la operación del Puerto de Chancay."
)

if st.button("⚡ Generar Plan de Intervención con IA"):
  with st.spinner("Procesando datos y consultando a Gemini 3.8 Flash..."):
    try:
      # Inicializar cliente conectando con los Secrets de Streamlit
      api_key = st.secrets["GEMINI_API_KEY"]
      client = genai.Client(api_key=api_key)

      # Armado del Payload JSON estructurado
      payload_operario = {
          "operario_id": trabajador["operario_id"],
          "probabilidad_cese_pct": float(trabajador["probabilidad_cese_pct"]),
          "nivel_riesgo": trabajador["nivel_riesgo"],
          "top_factores_riesgo": [
              {
                  "variable": "evaluacion_desempeno",
                  "valor_observado": float(trabajador["evaluacion_desempeno"]),
              },
              {
                  "variable": "es_temporada_alta_agricola",
                  "valor_observado": float(
                      trabajador["es_temporada_alta_agricola"]
                  ),
              },
              {
                  "variable": "dias_entrenamiento",
                  "valor_observado": float(trabajador["dias_entrenamiento"]),
              },
          ],
      }

      prompt_consulta = f"""
            Analiza los siguientes resultados numéricos del modelo predictivo y genera la recomendación prescriptiva para el supervisor de planta:

            DATOS DEL OPERARIO:
            {json.dumps(payload_operario, indent=2, ensure_ascii=False)}
            """

      # Solicitud a la API
      response = client.models.generate_content(
          model="gemini-3.8-flash", contents=prompt_consulta
      )

      st.success("Plan generado exitosamente.")
      st.markdown(response.text)

    except Exception as e:
      st.error(
          f"Error al conectar con la API de Gemini: {e}. Verifique la"
          " configuración de 'GEMINI_API_KEY' en Secrets."
      )
