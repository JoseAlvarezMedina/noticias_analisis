import os
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_topic_classifier(
    data_path: str,
    model_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
    C: float = 1.0,
    max_iter: int = 500
):
    """
    Entrena un clasificador de tópicos (TF-IDF + LogReg) y lo guarda en model_path.
    Devuelve el pipeline entrenado.
    """
    # 1) Cargar datos
    df = pd.read_csv(data_path)
    X = (df["titulo"].fillna("") + " " + df["resumen_clean_adv"])
    y = df["topic"]

    # 2) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 3) Pipeline
    pipe = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_df=0.8,
                    min_df=5,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    class_weight="balanced",
                    max_iter=max_iter,
                    random_state=random_state,
                ),
            ),
        ]
    )

    # 4) Entrenar
    pipe.fit(X_train, y_train)

    # 5) Evaluar
    y_pred = pipe.predict(X_test)
    print("=== Classification report ===")
    print(classification_report(y_test, y_pred))

    # 6) Guardar modelo
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipe, model_path)
    print(f"Modelo guardado en: {model_path}")

    return pipe

def load_topic_classifier(model_path: str):
    """Carga el modelo .pkl y lo devuelve."""
    return joblib.load(model_path)

def predict_topic(texts, model):
    """Predice el tópico para uno o varios textos."""
    if isinstance(model, str):
        model = load_topic_classifier(model)
    if isinstance(texts, str):
        texts = [texts]
    return model.predict(texts)

def predict_topic_proba(texts, model):
    """Devuelve la matriz de probabilidades por tópico."""
    if isinstance(model, str):
        model = load_topic_classifier(model)
    if isinstance(texts, str):
        texts = [texts]
    return model.predict_proba(texts)
