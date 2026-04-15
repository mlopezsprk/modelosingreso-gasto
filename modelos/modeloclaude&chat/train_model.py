import pandas as pd
import pickle
import re

from sentence_transformers import SentenceTransformer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report


# ── NORMALIZACIÓN ─────────────────────────────────────────

def normalizar(texto):
    texto = texto.lower()
    texto = re.sub(r'[€$£]', ' euros ', texto)
    texto = re.sub(r'\d+', ' __NUM__ ', texto)
    texto = re.sub(r'[^\w\s]', ' ', texto)
    return texto


# ── VERBOS ACCIONABLES ────────────────────────────────────
# Palabras que indican con certeza la dirección del dinero.
# Solo se usan para el filtro de calidad, no como reglas del modelo.

VERBOS_ACCIONABLES = {
    # gasto
    "pagué", "pagar", "pagado", "pagamos", "gasté", "gastado", "gastamos",
    "cobrado", "cobraron", "cobró", "cobrar", "desembolsé", "desembolsado",
    "aboné", "abonado", "solté", "costó", "costaron", "salió", "cargado",
    "cargaron", "cargó", "tuve", "acabo", "invertí",
    # ingreso
    "cobré", "recibí", "recibido", "ingresé", "ingresaron", "ingresado",
    "percibí", "percibido", "pagaron", "transfirieron", "transferido",
    "abonaron", "abonó", "cayeron", "llegaron", "llegó", "obtuve",
    "devolvieron", "devuelto", "entraron", "he cobrado", "me pagaron",
}

def tiene_verbo_accionable(texto):
    palabras = set(texto.lower().split())
    return bool(palabras & VERBOS_ACCIONABLES)


# ── CARGA Y LIMPIEZA DEL DATASET (opción 3) ───────────────

df = pd.read_csv("data/dataset_vosk.csv")
df = df.dropna(subset=["oracion", "etiqueta"])
df = df[df["etiqueta"].isin(["gasto", "ingreso"])].copy()

df["oracion_norm"] = df["oracion"].apply(normalizar)
df["longitud"]     = df["oracion_norm"].str.split().str.len()
df["tiene_verbo"]  = df["oracion"].apply(tiene_verbo_accionable)

total_original = len(df)

# Filtrar ruido puro: frases de menos de 3 palabras sin ningún verbo accionable.
# Las frases cortas CON verbo se conservan (tienen señal real).
df = df[(df["longitud"] >= 3) | df["tiene_verbo"]].copy()

print(f"Dataset original:  {total_original} filas")
print(f"Dataset limpio:    {len(df)} filas  "
      f"(eliminadas: {total_original - len(df)})")
print(f"Balance clases:    {df['etiqueta'].value_counts().to_dict()}")

X_texto = df["oracion_norm"].tolist()
y       = df["etiqueta"].tolist()


# ── EMBEDDING MODEL ───────────────────────────────────────

print("\nCargando modelo de embeddings...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

print("Generando embeddings...")
X_emb = embedder.encode(X_texto, show_progress_bar=True, batch_size=128)


# ── TRAIN / TEST ──────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X_emb, y, test_size=0.2, random_state=42, stratify=y
)


# ── MODELO: LinearSVC calibrado (opción 2) ────────────────
# LinearSVC encuentra un hiperplano de mayor margen que LogisticRegression
# en espacios de alta dimensión como los embeddings (384 dims).
# CalibratedClassifierCV añade probabilidades calibradas para la confianza.

modelo = CalibratedClassifierCV(
    LinearSVC(C=5, max_iter=3000),
    cv=3
)
modelo.fit(X_train, y_train)


# ── EVALUACIÓN ────────────────────────────────────────────

y_pred = modelo.predict(X_test)

print("\nReporte (test 20%):")
print(classification_report(y_test, y_pred))

# Validación cruzada para una estimación más robusta
print("Validación cruzada (5-fold, sobre muestra de 3000):")
idx = list(range(len(X_emb)))
import random; random.seed(42); random.shuffle(idx)
X_cv = X_emb[idx[:3000]]
y_cv = [y[i] for i in idx[:3000]]
cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores  = cross_val_score(modelo, X_cv, y_cv, cv=cv, scoring="f1_macro")
acc     = cross_val_score(modelo, X_cv, y_cv, cv=cv, scoring="accuracy")
print(f"  F1  por fold: {[f'{s:.3f}' for s in scores]}")
print(f"  F1  medio:    {scores.mean():.3f} ± {scores.std():.3f}")
print(f"  Acc medio:    {acc.mean():.3f} ± {acc.std():.3f}")


# ── GUARDAR ───────────────────────────────────────────────

with open("modelo.pkl", "wb") as f:
    pickle.dump((modelo, embedder), f)

print("\nModelo guardado como modelo.pkl")
