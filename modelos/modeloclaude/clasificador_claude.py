"""
Clasificador de Gastos e Ingresos — Vosk-ready
================================================
Uso:
    python clasificador.py

El modelo espera transcripciones de audio tal como las genera Vosk:
sin símbolos, importes en palabras ("trescientos euros", "mil doscientos").

Requisitos:
    pip install scikit-learn pandas num2words
"""

import os
import pickle
import re

# ── Stopwords ──────────────────────────────────────────────────────────────
# "me", "le", "te" NO están: indican la dirección del dinero.
#   "me pagaron" (ingreso) vs "he pagado" (gasto)
STOPWORDS = {
    "de","la","el","en","a","y","que","por","con","una","un","al","lo",
    "su","es","del","las","los","mi","este","esta","para","como","más",
    "pero","sus","ya","no","o","si","sobre","también","hasta","hay",
    "donde","desde","todo","durante","sido","fue","unos","unas","muy",
    "bien","cuando","ha","han","hoy","ayer","fin","poco","días","semana",
    "mes","año","mañana","tarde","noche","lunes","martes","miércoles",
    "jueves","viernes","enero","febrero","marzo","abril","mayo","junio",
    "julio","agosto","septiembre","octubre","noviembre","diciembre",
    "trimestre","quincena","pasado","pasada","próximo","próxima",
    "otro","otra","casi","alrededor","aproximadamente","total","final",
    "anoche","recientemente","pronto",
}

# Patrón de palabras numéricas en español
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
    """
    Preprocesa una transcripción Vosk para el clasificador.
    Reemplaza importes (en palabras o dígitos) por __NUM__
    y elimina palabras sin valor clasificatorio.
    """
    texto = texto.lower()

    # Normalizar símbolos de moneda por si acaso
    texto = re.sub(r'[€$£]', ' euros ', texto)

    # Reemplazar secuencias numéricas en palabras
    texto = _PATRON_NUM.sub(' __NUM__ ', texto)

    # Reemplazar dígitos residuales
    texto = re.sub(r'\d[\d.,]*\s*(?:euros?|dólares?|dolares?|libras?|pesos?)?',
                   ' __NUM__ ', texto)

    # Limpiar puntuación
    texto = re.sub(r'[^\w\s]', ' ', texto)

    # Eliminar stopwords
    tokens = [t for t in texto.split() if t and t not in STOPWORDS]
    return ' '.join(tokens)


def cargar_modelo(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


def predecir(modelo, oracion: str) -> dict:
    norm     = normalizar(oracion)
    etiqueta = modelo.predict([norm])[0]
    probs    = modelo.predict_proba([norm])[0]
    clases   = modelo.classes_
    return {
        'etiqueta':  etiqueta,
        'confianza': dict(zip(clases, [round(p * 100, 1) for p in probs])),
    }


def main():
    modelo_path = os.path.join(os.path.dirname(__file__), 'modelo.pkl')

    if not os.path.exists(modelo_path):
        print(f"Error: no se encontró '{modelo_path}'.")
        print("Asegúrate de que modelo.pkl está en la misma carpeta que este script.")
        return

    modelo = cargar_modelo(modelo_path)

    print("=" * 50)
    print("  [MODELO: claude]")
    print("  Clasificador de gastos e ingresos")
    print("  Escribe la transcripción del audio.")
    print("  'salir' para terminar.")
    print("=" * 50)

    while True:
        try:
            oracion = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nHasta luego.")
            break

        if not oracion:
            continue
        if oracion.lower() in {"salir", "exit", "quit"}:
            print("Hasta luego.")
            break

        r     = predecir(modelo, oracion)
        icono = "💸" if r['etiqueta'] == "gasto" else "💰"

        print(f"\n  {icono}  {r['etiqueta'].upper()}")
        print(f"     Gasto:   {r['confianza'].get('gasto', 0):.1f}%")
        print(f"     Ingreso: {r['confianza'].get('ingreso', 0):.1f}%")


if __name__ == "__main__":
    main()
