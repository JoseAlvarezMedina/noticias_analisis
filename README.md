
# Observatorio de Opinión Corporativa Colombiana

Un dashboard interactivo construido con Streamlit para explorar y analizar noticias locales en Colombia a través de:

- **KPIs**: número de noticias, titulares únicos y tópicos distintos.  
- **Sentimiento**: velocímetro (gauge) que muestra el sentimiento medio filtrable por medio o tópico.  
- **Proporción de tópicos**: gráfico de dona con la distribución porcentual de los temas.  
- **¿Qué pasó hoy? (RAG)**: interfaz de pregunta-respuesta que resume en dos líneas los artículos más relevantes usando embeddings y GPT-4o-mini.  
- **Social Listening**: grafo interactivo (PyVis) que muestra relaciones entre medios, organizaciones y personas, filtrado por el tópico dominante y con detalle de las noticias que conectan cada par de nodos.

---

## 📁 Estructura

```

.
├── .streamlit/
│   ├── config.toml       ← configuración de tema y layout
│   └── secrets.toml      ← tu OPENAI\_API\_KEY
├── app/
│   └── app.py            ← aplicación Streamlit principal
├── data/
│   └── processed/
│       └── noticias\_with\_entities.csv   ← datos procesados con entidades
├── models/
│   ├── topic\_classifier.pkl              ← modelo de clasificación de tópicos
│   ├── embeddings.pkl                    ← embeddings + metadatos (RAG)
│   └── graph.pkl                         ← grafo de relaciones (pickle)
├── requirements.txt     ← dependencias de Python
└── src/
├── build\_embeddings.py  ← genera embeddings FAISS
├── features.py          ← feature-engineering básico
├── features\_graph.py    ← extrae entidades y arma el grafo
├── modeling.py          ← entrena el clasificador de tópicos
├── train\_topic.py       ← wrapper para entrenar topic model
└── utils.py             ← utilidades de carga y parsing

````

---

## 🛠️ Instalación

1. **Clona** este repositorio y sitúate en la raíz:
   ```bash
   git clone https://github.com/JoseAlvarezMedina/noticias_analisis.git
   cd noticias_analisis
````

2. **Crea** un entorno virtual e instala dependencias:

   ```bash
   python -m venv .venv
   source .venv/bin/activate    # Linux/Mac
   .\.venv\Scripts\activate     # Windows
   pip install -r requirements.txt
   ```

3. **Configura** tu clave de OpenAI en `.streamlit/secrets.toml`:

   ```toml
   [general]
   OPENAI_API_KEY = "sk-··············"
   ```

---

## ▶️ Cómo ejecutar

```bash
streamlit run app/app.py
```

Abre tu navegador en la URL que Streamlit muestre (por defecto [http://localhost:8501](http://localhost:8501)).

---

## 🔍 Flujo de la aplicación

1. **Filtros**: rango de fechas, selección de medios y filtro de sentimiento.
2. **KPIs**: vistas rápidas de volumen, unicidad y variedad de tópicos.
3. **Sentimiento & Tópicos**: gauge interactivo + dona de tópicos en una misma fila.
4. **¿Qué pasó hoy?**: formular pregunta en lenguaje natural y obtener un resumen automatizado.
5. **Social Listening**:

   * Selecciona una entidad (organización/persona).
   * Descubre su tópico dominante.
   * Explora el sub-grafo de conexiones y ve el detalle de las noticias que unen cada par de nodos.

---

## 🛠️ Tecnologías

* **Streamlit**: interfaz web
* **Plotly**: velocímetro y gráficas
* **PyVis & NetworkX**: visualización de grafos
* **scikit-learn**: clasificación de tópicos
* **sentence-transformers + FAISS**: embeddings semánticos
* **OpenAI GPT-4o-mini**: generación de resúmenes RAG

---

## 🎯 Casos de uso

* Monitoreo de reputación corporativa en medios digitales.
* Detección temprana de crisis o riesgos regulatorios.
* Soporte a la toma de decisiones basado en análisis cuantitativo y cualitativo de noticias.

---

> *Desarrollado por José Álvarez · [Mi LinkedIn](https://www.linkedin.com/in/JoseAlvarezMedina)*

