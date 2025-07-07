def to_label(score: float, pos_thr: float = 0.05, neg_thr: float = -0.05) -> str:
    """
    Convierte un score numérico de sentimiento en etiqueta categórica:
      score >= pos_thr → 'positive'
      score <= neg_thr → 'negative'
      otherwise       → 'neutral'
    """
    if score >= pos_thr:
        return "positive"
    if score <= neg_thr:
        return "negative"
    return "neutral"
