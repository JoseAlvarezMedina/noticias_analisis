from pathlib import Path
from src.modeling import train_topic_classifier

DATA_PATH  = Path("data/processed/noticias_full_processed.csv")
MODEL_PATH = Path("models/topic_classifier.pkl")

if __name__ == "__main__":
    # ⚠️ pasa argumentos en orden, sin el nombre
    train_topic_classifier(
        str(DATA_PATH),        # 1-er argumento: ruta CSV
        str(MODEL_PATH),       # 2-o argumento: ruta pkl
        0.2,                   # test_size
        42,                    # random_state
        1.0,                   # C
        500                    # max_iter
    )
