import numpy as np
import textstat
import tldextract
from .cleaning import clean_text

def add_features(df):
    """
    Añade columnas derivadas:
      - medio      : dominio extraído de la URL
      - autor      : rellena nulos con 'Desconocido'
      - resumen_clean_adv : texto limpio (clean_text)
      - word_count : conteo de palabras del texto limpio
      - readability: índice de legibilidad Flesch reading ease
    Devuelve un DataFrame copiado con estas nuevas columnas.
    """
    df = df.copy()
    # Medio (dominio)
    def _get_domain(url):
        ext = tldextract.extract(url)
        if ext.domain and ext.suffix:
            return f"{ext.domain}.{ext.suffix}"
        return url
    df["medio"] = df["url"].map(_get_domain)

    # Autor por defecto
    df["autor"] = df["autor"].fillna("Desconocido")

    # Texto limpio y métricas
    df["resumen_clean_adv"] = df["resumen"].map(clean_text)
    df["word_count"] = df["resumen_clean_adv"].str.split().str.len()
    df["readability"] = df["resumen_clean_adv"].apply(
        lambda txt: textstat.flesch_reading_ease(txt) if txt else np.nan
    )

    return df
