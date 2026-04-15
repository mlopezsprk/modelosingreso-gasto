import pandas as pd
import pickle
import re

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# ── NORMALIZACIÓN ─────────────────────────────────────────

def normalizar(texto):
    texto = texto.lower()
    texto = re.sub(r'[€$£]', ' euros ', texto)
    texto = re.sub(r'\d+', ' __NUM__ ', texto)
    texto = re.sub(r'[^\w\s]', ' ', texto)
    return texto


# ── CARGA DATASET ─────────────────────────────────────────

df = pd.read_csv("data/dataset_vosk.csv")

X = df["oracion"].apply(normalizar).tolist()
y = df["etiqueta"].tolist()


# ── EMBEDDING MODEL ───────────────────────────────────────

print("Cargando modelo de embeddings...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

print("Generando embeddings...")
X_emb = embedder.encode(X, show_progress_bar=True)


# ── TRAIN / TEST ──────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X_emb, y, test_size=0.2, random_state=42
)


# ── MODELO ────────────────────────────────────────────────

modelo = LogisticRegression(max_iter=2000)
modelo.fit(X_train, y_train)


# ── EVALUACIÓN ────────────────────────────────────────────

y_pred = modelo.predict(X_test)

print("\nReporte:")
print(classification_report(y_test, y_pred))


# ── GUARDAR ───────────────────────────────────────────────

with open("modelo.pkl", "wb") as f:
    pickle.dump((modelo, embedder), f)

print("\nModelo guardado como modelo.pkl")