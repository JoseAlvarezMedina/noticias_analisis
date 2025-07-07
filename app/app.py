# app/app.py
import pandas as pd
import altair as alt
import streamlit as st
import joblib, pickle
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

import plotly.graph_objects as go        # gauge
import plotly.express   as px            # barras & pie

# ─────────────────────────────────────────────────────────────
# 1. Carga de datos, modelos y embeddings
# ─────────────────────────────────────────────────────────────
DATA_PATH = Path("data/processed/noticias_full_processed.csv")
MODEL_PATH = Path("models/topic_classifier.pkl")
EMB_PATH  = Path("models/embeddings.pkl")   # generado por build_embeddings.py


@st.cache_data(show_spinner=False)
def load_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["fecha"])


@st.cache_resource(show_spinner=False)
def load_topic_model(path: Path):
    return joblib.load(path)


@st.cache_resource(show_spinner=False)
def load_vector_store(path: Path):
    X, meta, encoder_name = pickle.load(path.open("rb"))
    encoder = SentenceTransformer(encoder_name)
    return X, meta, encoder


df_full   = load_dataset(DATA_PATH)
df_unique = df_full.drop_duplicates(subset="titulo")
topic_model            = load_topic_model(MODEL_PATH)
X_vec, df_meta, encoder = load_vector_store(EMB_PATH)

# ─────────────────────────────────────────────────────────────
# 2. Sidebar · filtros
# ─────────────────────────────────────────────────────────────
st.sidebar.header("Filtros")

use_unique = st.sidebar.checkbox(
    "Usar solo títulos únicos", False, help="Quita titulares repetidos."
)
df_base = df_unique if use_unique else df_full

# Fechas
min_date, max_date = df_base["fecha"].min().date(), df_base["fecha"].max().date()
date_range = st.sidebar.date_input(
    "Rango de fechas", (min_date, max_date),
    min_value=min_date, max_value=max_date
)

# Medios
all_medios = sorted(df_base["medio"].unique())
sel_medios = st.sidebar.multiselect("Medios", all_medios, default=all_medios)

# Filtro avanzado de sentimiento
with st.sidebar.expander("Filtros avanzados"):
    sent_min, sent_max = st.slider(
        "Sentimiento (–1 = Neg ● 1 = Pos)",
        -1.0, 1.0, (-1.0, 1.0), step=0.05
    )

# Aplicar filtros
start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
mask = (
    (df_base["fecha"].between(start_date, end_date)) &
    (df_base["medio"].isin(sel_medios)) &
    (df_base["sent_vader"].between(sent_min, sent_max))
)
df_filt = df_base.loc[mask].copy()

# ─────────────────────────────────────────────────────────────
# 3. KPIs
# ─────────────────────────────────────────────────────────────
st.title("Dashboard de Noticias Colombianas")

unique_titles = df_filt["titulo"].nunique()
c1, c2, c3 = st.columns(3)
c1.metric("Noticias",          f"{len(df_filt):,}")
c2.metric("Titulares únicos",  f"{unique_titles:,}")
c3.metric("Tópicos distintos", df_filt["topic"].nunique())

# ─────────────────────────────────────────────────────────────
# 4. Sentimiento: gauge + barras
# ─────────────────────────────────────────────────────────────
st.subheader("Sentimiento general / comparativo")

# ---- selección medio / tópico -------------------------------
group_choice = st.radio(
    "Agrupar sentimiento por:",
    ["Medio", "Tópico"],
    horizontal=True,
)

if group_choice == "Medio":
    sent_grp = (
        df_filt.groupby("medio")["sent_vader"].mean().reset_index()
        .sort_values("sent_vader")
    )
    label_col = "medio"
else:
    sent_grp = (
        df_filt.groupby("topic")["sent_vader"].mean().reset_index()
        .sort_values("sent_vader")
    )
    label_col = "topic"

# ---- selectbox para el gauge -------------------------------
entity_selected = st.selectbox(
    f"Selecciona un {group_choice.lower()} para el velocímetro:",
    sent_grp[label_col],
    index=0,
)

value_selected = sent_grp.loc[
    sent_grp[label_col] == entity_selected, "sent_vader"
].values[0]

def gauge(value: float):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"font":{"size":28}},
            gauge={
                "axis": {"range": [-1, 1]},
                "bar":  {"color": "rgba(0,0,0,0)"},
                "steps": [
                    {"range": [-1, -0.2], "color": "#FF6B6B"},
                    {"range": [-0.2, 0.2], "color": "#FFD93B"},
                    {"range": [0.2, 1], "color": "#18CE66"},
                ],
                "threshold": {
                    "line": {"color": "royalblue", "width": 6},
                    "thickness": 0.9,
                    "value": value,
                },
            },
        )
    )
    return fig.update_layout(margin=dict(t=10,b=0,l=0,r=0))

st.plotly_chart(gauge(value_selected), use_container_width=True)

# ---- barras horizontales normalizadas ----------------------
max_abs = sent_grp["sent_vader"].abs().max()
fig_bar = px.bar(
    sent_grp,
    x="sent_vader",
    y=label_col,
    orientation="h",
    color="sent_vader",
    color_continuous_scale=["#FF6B6B", "#FFD93B", "#18CE66"],
    range_x=[-max_abs, max_abs],
    labels={"sent_vader": "sentimiento", label_col: ""},
)
fig_bar.update_layout(coloraxis_showscale=False, height=300, margin=dict(t=10,l=0,r=0,b=0))
st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────────────────────
# 5. Distribución de tópicos (torta)
# ─────────────────────────────────────────────────────────────
st.subheader("Proporción de tópicos")

topic_count = df_filt["topic"].value_counts().reset_index()
topic_count.columns = ["topic", "count"]

fig_pie = px.pie(
    topic_count, names="topic", values="count",
    color="topic", hole=.35,
    title="",
)
fig_pie.update_layout(legend_orientation="h", legend_y=-0.1, margin=dict(t=20,b=0,l=0,r=0))
st.plotly_chart(fig_pie, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────────────────────
# 6. Buscador de noticias
# ─────────────────────────────────────────────────────────────
st.subheader("Buscador de noticias")

keyword = st.text_input("Escribe una palabra o frase:")

if keyword:
    kw_mask = df_filt["titulo"].str.contains(keyword, case=False, na=False) | \
              df_filt["resumen"].str.contains(keyword, case=False, na=False)
    results = df_filt.loc[kw_mask, ["fecha", "medio", "titulo", "resumen", "url"]]
    st.write(f"Se encontraron **{len(results)}** coincidencias.")
    st.dataframe(results, hide_index=True, use_container_width=True)
else:
    st.info("Introduce una palabra para buscar dentro del conjunto filtrado.")

st.divider()

# ──────────────────────────
# 7. Q&A · RAG — “¿Qué pasó hoy?”
# ──────────────────────────
st.header("¿Qué pasó hoy? Pregunta aquí")

question = st.text_input("Haz tu pregunta:")
top_k     = st.slider("Noticias a considerar", 3, 10, 5)

if st.button("Responder", key="btn-rag"):
    if not question.strip():
        st.warning("Escribe una pregunta primero.")
    else:
        q_emb  = encoder.encode([question])
        sims   = cosine_similarity(q_emb, X_vec).flatten()
        hits   = df_meta.iloc[sims.argsort()[-top_k:][::-1]]

        context = "\n\n".join(
            f"[{r.medio}] {r.titulo} — {r.resumen or ''}" for r in hits.itertuples()
        )

        if "OPENAI_API_KEY" not in st.secrets:
            st.warning("Añade tu OPENAI_API_KEY en .streamlit/secrets.toml")
        else:
            client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
            prompt = (
                "Eres un analista de medios. Con base en las noticias listadas, "
                "responde en ≤120 palabras y en español:\n\n"
                f"{context}\n\nPregunta: {question}\nRespuesta:"
            )
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
            )
            st.success(resp.choices[0].message.content)

            with st.expander("Fuentes verificadas"):
                st.dataframe(
                    hits[["fecha", "medio", "titulo", "url"]],
                    hide_index=True, use_container_width=True
                )

st.divider()

# ──────────────────────────
# 8. Clasificador de tópico en vivo
# ──────────────────────────
st.subheader("Clasifica una noticia al instante")

user_text = st.text_area(
    "Pega aquí el **título + resumen** y haz clic en “Clasificar”:", height=120
)

if st.button("Clasificar", key="btn-predict-topic"):
    if user_text.strip():
        pred = topic_model.predict([user_text])[0]
        st.success(f"Tópico sugerido: **{pred}**")
    else:
        st.warning("Introduce texto para clasificar.")

st.caption(
    "Embeddings = MiniLM · Similitud coseno · LLM = gpt-4o-mini · "
    "Modelo de tópicos = Regresión Logística (93 % precisión)"
)
