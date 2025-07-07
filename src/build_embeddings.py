from pathlib import Path
import pandas as pd, numpy as np, pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

DATA_CSV  = Path("data/processed/noticias_full_processed.csv")
OUT_PATH  = Path("models/embeddings.pkl")   # (matrix, df_meta, encoder_name)

print("▶ Leyendo CSV y quitando títulos duplicados…")
df = (
    pd.read_csv(DATA_CSV, parse_dates=["fecha"])
      .drop_duplicates(subset="titulo")
      .reset_index(drop=True)
)

print("▶ Generando embeddings (MiniLM)…")
ENCODER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model   = SentenceTransformer(ENCODER)
X       = model.encode(df["titulo"] + " " + df["resumen"].fillna(""), show_progress_bar=True)

print("▶ Guardando matriz + metadatos…")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
pickle.dump((X.astype("float32"), df, ENCODER), OUT_PATH.open("wb"))
print("✅ Embeddings guardados en", OUT_PATH)
