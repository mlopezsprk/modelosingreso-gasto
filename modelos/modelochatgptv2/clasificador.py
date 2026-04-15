import os
import pickle
import re

# ── STOPWORDS REDUCIDAS ─────────────────────────────────────
# Quitamos menos cosas → mantenemos más señal
STOPWORDS = {
    "de","la","el","en","a","y","que","por","con","una","un","al",
    "del","las","los","este","esta","para","como","más",
    "pero","ya","no","o","si","sobre","también","hasta",
    "donde","desde","todo","durante","muy","bien"
}

# ── PATRÓN NUMÉRICO ─────────────────────────────────────────
_NUM = (
    r'\b(?:cero|un[ao]?|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|'
    r'diez|once|doce|trece|catorce|quince|diecis[eé]is|diecisiete|'
    r'dieciocho|diecinueve|veinte|treinta|cuarenta|cincuenta|'
    r'sesenta|setenta|ochenta|noventa|cien(?:to)?|doscient[ao]s|'
    r'trescient[ao]s|cuatrocient[ao]s|quinient[ao]s|seiscient[ao]s|'
    r'setecient[ao]s|ochocient[ao]s|novecient[ao]s|mil(?:es)?)\b'
)

_PATRON_NUM = re.compile(_NUM, re.IGNORECASE)


def normalizar(texto: str) -> str:
    texto = texto.lower()

    # ── 1. Normalizar moneda
    texto = re.sub(r'[€$£]', ' euros ', texto)

    # ── 2. Reemplazar números
    texto = _PATRON_NUM.sub(' __NUM__ ', texto)
    texto = re.sub(r'\d[\d.,]*', ' __NUM__ ', texto)

    # ── 3. Limpiar puntuación
    texto = re.sub(r'[^\w\s]', ' ', texto)

    # ── 4. Tokens
    tokens = texto.split()

    # ── 5. Eliminar stopwords (menos agresivo)
    tokens = [t for t in tokens if t not in STOPWORDS]

    # ── 6. FEATURE CLAVE (sin hardcodear reglas)
    # Detectar dirección del dinero como token extra
    if "me" in tokens and ("pagaron" in tokens or "ingresaron" in tokens or "abonaron" in tokens):
        tokens.append("__INGRESO__")

    if ("pague" in tokens or "pagado" in tokens or "gaste" in tokens or "cobraron" in tokens):
        tokens.append("__GASTO__")

    return ' '.join(tokens)


def cargar_modelo(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


def predecir(modelo, oracion: str) -> dict:
    norm = normalizar(oracion)

    etiqueta = modelo.predict([norm])[0]
    probs = modelo.predict_proba([norm])[0]
    clases = modelo.classes_

    return {
        'etiqueta': etiqueta,
        'confianza': dict(zip(clases, [round(p * 100, 1) for p in probs])),
        'texto_normalizado': norm  # 🔥 útil para debug
    }


def main():
    modelo_path = os.path.join(os.path.dirname(__file__), 'modelo.pkl')

    if not os.path.exists(modelo_path):
        print(f"Error: no se encontró '{modelo_path}'.")
        return

    modelo = cargar_modelo(modelo_path)

    print("=" * 50)
    print("  Clasificador de gastos e ingresos (mejorado)")
    print("=" * 50)

    while True:
        try:
            oracion = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not oracion:
            continue
        if oracion.lower() in {"salir", "exit"}:
            break

        r = predecir(modelo, oracion)
        icono = "💸" if r['etiqueta'] == "gasto" else "💰"

        print(f"\n{icono} {r['etiqueta'].upper()}")
        print(f"Gasto:   {r['confianza'].get('gasto', 0):.1f}%")
        print(f"Ingreso: {r['confianza'].get('ingreso', 0):.1f}%")

        # 🔥 DEBUG PRO
        print(f"(debug) → {r['texto_normalizado']}")


if __name__ == "__main__":
    main()