import pandas as pd
from bs4 import BeautifulSoup
import re
import spacy

# Carga y configuración de spaCy
nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])
STOPWORDS = nlp.Defaults.stop_words

def load_data(path: str) -> pd.DataFrame:
    """
    Carga el CSV y parsea la columna 'fecha' a datetime.
    """
    return pd.read_csv(path, parse_dates=["fecha"])

def clean_text(text: str) -> str:
    """
    Limpieza avanzada de un texto:
      - Elimina HTML
      - Quita URLs
      - Pasa a minúsculas y elimina caracteres no alfabéticos
      - Lematiza y elimina stopwords
    """
    if pd.isna(text):
        return ""
    # 1) Eliminar HTML
    txt = BeautifulSoup(text, "lxml").get_text()
    # 2) Quitar URLs
    txt = re.sub(r"http\S+", "", txt)
    # 3) Minúsculas y filtrar caracteres
    txt = re.sub(r"[^a-záéíóúñü\s]", " ", txt.lower())
    # 4) Lematizar y quitar stopwords
    doc = nlp(txt)
    tokens = [
        token.lemma_
        for token in doc
        if token.lemma_ not in STOPWORDS and len(token.lemma_) > 2
    ]
    return " ".join(tokens)
