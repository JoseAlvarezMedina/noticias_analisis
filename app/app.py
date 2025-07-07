# app/app.py
import pandas as pd
import altair as alt
import streamlit as st
import joblib
from pathlib import Path

# ────────────────────────────────────────────────────────────────────────────────
# 1. Utilidades de carga ────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────
DATA_PATH  = Path("data/processed/noticias_full_processed.csv")
MODEL_PATH = Path("models/topic_classifier.pkl")

@st.cache_data(show_spinner=False)
def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["fecha"])
    return df

@st.cache_resource(show_spinner=False)
def load_model(path: Path):
    return joblib.load(path)

df = load_dataset(DATA_PATH)
model = load_model(MODEL_PATH)

# ────────────────────────────────────────────────────────────────────────────────
# 2. Sidebar: filtros ───────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────
st.sidebar.header("Filtros")

# Rango de fechas
min_date, max_date = df["fecha"].min().date(), df["fecha"].max().date()
date_range = st.sidebar.date_input(
    "Rango de fechas",
    (min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Medios
all_medios = sorted(df["medio"].unique())
sel_medios = st.sidebar.multiselect("Medios", all_medios, default=all_medios)

# Sentimiento VADER
sent_min, sent_max = st.sidebar.slider(
    "Filtro Sentimiento VADER",
    -1.0, 1.0,
    (-1.0, 1.0),
    step=0.05
)

# Aplicar filtros
start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
mask = (
    (df["fecha"].between(start_date, end_date)) &
    (df["medio"].isin(sel_medios)) &
    (df["sent_vader"].between(sent_min, sent_max))
)
df_filt = df.loc[mask].copy()

# ────────────────────────────────────────────────────────────────────────────────
# 3. KPIs ────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────
st.title("Dashboard de Noticias Colombianas")

col1, col2, col3 = st.columns(3)
col1.metric("Noticias filtradas", f"{len(df_filt):,}")
col2.metric("Positivas (VADER)", f"{(df_filt['sent_vader_cat']=='positive').mean()*100:.1f}%")
col3.metric("Tópicos únicos", df_filt["topic"].nunique())

st.divider()

# ────────────────────────────────────────────────────────────────────────────────
# 4. Gráficos ────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────
st.subheader("Distribución de tópicos")
topic_count = (
    df_filt["topic"]
    .value_counts(normalize=True)
    .rename("pct")
    .reset_index()
    .rename(columns={"index": "topic"})
)
chart_topic = (
    alt.Chart(topic_count)
    .mark_bar()
    .encode(
        x=alt.X("pct:Q", axis=alt.Axis(format="%")),
        y=alt.Y("topic:N", sort="-x"),
        tooltip=["topic", alt.Tooltip("pct", format=".1%")]
    )
)
st.altair_chart(chart_topic, use_container_width=True)

st.subheader("Volumen semanal")
df_filt["week"] = df_filt["fecha"].dt.to_period("W").dt.start_time
weekly = (
    df_filt.groupby("week")["url"]
    .size()
    .reset_index(name="count")
)
chart_week = (
    alt.Chart(weekly)
    .mark_line(point=True)
    .encode(
        x="week:T",
        y="count:Q",
        tooltip=["week:T", "count:Q"]
    )
)
st.altair_chart(chart_week, use_container_width=True)

st.divider()

# ────────────────────────────────────────────────────────────────────────────────
# 5. Clasificación en vivo ──────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────
st.header("Clasificador de tópico en vivo")

user_text = st.text_area(
    "Pega aquí el **título + resumen** de una noticia y presiona “Predecir”:",
    height=120
)

if st.button("Predecir"):
    if user_text.strip():
        pred = model.predict([user_text])[0]
        st.success(f"**Tópico predicho:** {pred}")
    else:
        st.warning("Introduce texto para clasificar.")

st.caption("Modelo: TF-IDF + LogisticRegression · Accuracy 93 %")
