"""
Entrenamiento del clasificador Gasto/Ingreso
=============================================
Optimiza el modelo para precisión > 75% usando solo ML, sin edge cases.

Uso:
    python train_model.py
"""

import os
import sys
import re
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ── Importar normalizar del certificador ──────────────────────────────────────
# (Copiamos aquí para evitar dependencias circulares)

STOPWORDS = {
    "de", "la", "el", "en", "a", "y", "que", "por", "con", "una", "un", "al", "lo",
    "su", "es", "del", "las", "los", "mi", "este", "esta", "para", "como", "más",
    "pero", "sus", "ya", "no", "o", "si", "sobre", "también", "hasta", "hay",
    "donde", "desde", "todo", "durante", "sido", "fue", "unos", "unas", "muy",
    "bien", "cuando", "ha", "han", "hoy", "ayer", "fin", "poco", "días", "semana",
    "mes", "año", "mañana", "tarde", "noche", "lunes", "martes", "miércoles",
    "jueves", "viernes", "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
    "trimestre", "quincena", "pasado", "pasada", "próximo", "próxima",
    "otro", "otra", "casi", "alrededor", "aproximadamente", "total", "final",
    "anoche", "recientemente", "pronto",
}

_NUM = (
    r'\b(?:cero|un[ao]?|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|'
    r'diez|once|doce|trece|catorce|quince|diecis[eé]is|diecisiete|'
    r'dieciocho|diecinueve|veinte|veinti(?:ún|uno|una|dós|dos|trés|tres|'
    r'cuatro|cinco|séis|seis|siete|ocho|nueve)|'
    r'treinta|cuarenta|cincuenta|sesenta|setenta|ochenta|noventa|'
    r'cien(?:to)?|doscient[ao]s|trescient[ao]s|cuatrocient[ao]s|'
    r'quinient[ao]s|seiscient[ao]s|setecient[ao]s|ochocient[ao]s|'
    r'novecient[ao]s|mil(?:es)?|mill[oó]n(?:es)?)\b'
)
_MONEDA = r'(?:\s+(?:euros?|dólares?|dolares?|libras?|pesos?))?'
_PATRON_NUM = re.compile(
    _NUM + r'(?:\s+(?:y\s+)?' + _NUM + r')*'
    r'(?:\s+(?:con\s+)?' + _NUM + r'(?:\s+' + _NUM + r')*)?' + _MONEDA,
    re.IGNORECASE,
)


def normalizar(texto: str) -> str:
    """Normaliza texto transcrito con Vosk."""
    texto = texto.lower()
    texto = re.sub(r'[€$£]', ' euros ', texto)
    texto = _PATRON_NUM.sub(' __NUM__ ', texto)
    texto = re.sub(r'\d[\d.,]*\s*(?:euros?|dólares?|dolares?|libras?|pesos?)?',
                   ' __NUM__ ', texto)
    texto = re.sub(r'[^\w\s]', ' ', texto)
    tokens = [t for t in texto.split() if t and t not in STOPWORDS]
    return ' '.join(tokens)


def main():
    # ── Cargar dataset ────────────────────────────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "dataset_vosk.csv")
    
    print(f"Cargando dataset desde: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    print(f"Dataset cargado: {len(df)} muestras")
    print(f"Distribución: {df['etiqueta'].value_counts().to_dict()}")
    
    # ── Normalizar dataset ────────────────────────────────────────────────────
    print("\nNormalizando dataset...")
    X = df['oracion'].apply(normalizar)
    y = df['etiqueta']
    
    # ── Dividir datos (80/20) para evaluación ─────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Entrenamiento: {len(X_train)} | Prueba: {len(X_test)}")
    
    # ── Crear pipeline optimizado ─────────────────────────────────────────────
    print("\nEntrenando pipeline...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            min_df=1,
            max_df=1.0,
            lowercase=True,
            stop_words=None,  # Ya filtrados en normalizar()
        )),
        ('classifier', LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight='balanced',
            random_state=42,
            solver='lbfgs',
        )),
    ])
    
    # Entrenar en todos los datos de entrenamiento
    pipeline.fit(X_train, y_train)
    
    # ── Evaluar con cross-validation ──────────────────────────────────────────
    print("\nEvaluación con Cross-Validation (5-fold):")
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1_weighted')
    print(f"Scores: {[f'{s:.1%}' for s in cv_scores]}")
    print(f"Media: {cv_scores.mean():.1%} (+/- {cv_scores.std():.1%})")
    
    # ── Evaluar en test set ───────────────────────────────────────────────────
    print("\nEvaluación en Test Set:")
    train_acc = pipeline.score(X_train, y_train)
    test_acc = pipeline.score(X_test, y_test)
    print(f"Precisión Entrenamiento: {train_acc:.1%}")
    print(f"Precisión Test: {test_acc:.1%}")
    
    if test_acc >= 0.75:
        print(f"\n✅ Modelo alcanza precisión > 75%: {test_acc:.1%}")
    else:
        print(f"\n⚠️  Precisión aún baja: {test_acc:.1%} (objetivo: > 75%)")
    
    print("\nReporte clasificación (Test Set):")
    print(classification_report(y_test, pipeline.predict(X_test)))
    
    # ── Guardar modelo ───────────────────────────────────────────────────────
    model_path = os.path.join(script_dir, "modelo.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    
    print(f"\n✅ Modelo guardado en: {model_path}")


if __name__ == "__main__":
    main()
