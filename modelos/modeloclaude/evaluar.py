import os
import pickle
import re
import random
from collections import defaultdict

# ── NORMALIZADOR (MISMO QUE MODELO) ─────────────────────────

STOPWORDS = {
    "de","la","el","en","a","y","que","por","con","una","un","al",
    "del","las","los","este","esta","para","como","más",
    "pero","ya","no","o","si","sobre","también","hasta"
}

_NUM = r'\b(?:cero|uno|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|veinte|treinta|cuarenta|cincuenta|cien|doscientos|trescientos|mil)\b'
_PATRON_NUM = re.compile(_NUM, re.IGNORECASE)

def normalizar(texto):
    texto = texto.lower()
    texto = re.sub(r'[€$£]', ' euros ', texto)
    texto = _PATRON_NUM.sub(' __NUM__ ', texto)
    texto = re.sub(r'\d+', ' __NUM__ ', texto)
    texto = re.sub(r'[^\w\s]', ' ', texto)
    tokens = [t for t in texto.split() if t not in STOPWORDS]
    return ' '.join(tokens)


# ── FRASES POR DIFICULTAD ───────────────────────────────────

FRASES = {
    "facil": [
        ("me pagaron cien euros", "ingreso"),
        ("he pagado cincuenta euros", "gasto"),
    ],

    "media": [
        ("me entraron doscientos euros", "ingreso"),
        ("se me fueron cien euros", "gasto"),
    ],

    "dificil": [
        ("transferencia de dinero", "gasto"),
        ("me han pasado dinero", "ingreso"),
    ],

    "ruido": [
        ("me pagaro cien", "ingreso"),
        ("me cobraro cincuenta", "gasto"),
        ("me entro pasta", "ingreso"),
        ("me quitaron dinero", "gasto"),
    ]
}


# ── GENERADOR ADICIONAL ─────────────────────────────────────

VERBOS_GASTO = ["pague", "me cobraron", "gaste", "se me fueron"]
VERBOS_INGRESO = ["me pagaron", "me ingresaron", "me dieron"]

CANTIDADES = ["cien", "doscientos", "trescientos", "mil"]

def generar_extra(n=300):
    frases = []
    for _ in range(n):
        if random.random() < 0.5:
            frases.append((f"{random.choice(VERBOS_GASTO)} {random.choice(CANTIDADES)}", "gasto"))
        else:
            frases.append((f"{random.choice(VERBOS_INGRESO)} {random.choice(CANTIDADES)}", "ingreso"))
    return frases

FRASES["synthetic"] = generar_extra(300)


# ── EVALUACIÓN ─────────────────────────────────────────────

def evaluar_bloque(modelo, frases):
    correctos = 0
    total = len(frases)
    confs = []
    errores = []

    for frase, esperado in frases:
        norm = normalizar(frase)

        pred = modelo.predict([norm])[0]
        probs = modelo.predict_proba([norm])[0]
        clases = list(modelo.classes_)
        conf = probs[clases.index(pred)] * 100

        confs.append(conf)

        if pred == esperado:
            correctos += 1
        else:
            errores.append((frase, esperado, pred, conf))

    precision = correctos / total * 100
    conf_media = sum(confs) / len(confs)

    return precision, conf_media, errores


# ── MAIN ───────────────────────────────────────────────────

def main():
    modelo_path = os.path.join(os.path.dirname(__file__), 'modelo.pkl')

    with open(modelo_path, 'rb') as f:
        modelo = pickle.load(f)

    print("\n" + "=" * 60)
    print("EVALUACIÓN AVANZADA DEL MODELO")
    print("=" * 60)

    resultados = {}

    for categoria, frases in FRASES.items():
        precision, conf, errores = evaluar_bloque(modelo, frases)

        resultados[categoria] = {
            "precision": precision,
            "confianza": conf,
            "errores": errores
        }

        print(f"\n[{categoria.upper()}]")
        print(f"Precisión: {precision:.1f}%")
        print(f"Confianza media: {conf:.1f}%")

        if errores:
            print(f"Errores ({len(errores)}):")
            for e in errores[:5]:  # solo primeros 5
                print(f"  ❌ {e[0]} → {e[2]} ({e[3]:.1f}%)")

    # ── RESUMEN GLOBAL ─────────────────────────────────────

    total_prec = sum(r["precision"] for r in resultados.values()) / len(resultados)
    total_conf = sum(r["confianza"] for r in resultados.values()) / len(resultados)

    print("\n" + "=" * 60)
    print("RESUMEN GLOBAL")
    print("=" * 60)
    print(f"Precisión media: {total_prec:.1f}%")
    print(f"Confianza media: {total_conf:.1f}%")


if __name__ == "__main__":
    main()