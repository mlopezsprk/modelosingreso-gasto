import os
import pickle
import re


def normalizar(texto):
    texto = texto.lower()
    texto = re.sub(r'[€$£]', ' euros ', texto)
    texto = re.sub(r'\d+', ' __NUM__ ', texto)
    texto = re.sub(r'[^\w\s]', ' ', texto)
    return texto


def cargar_modelo(path):
    with open(path, "rb") as f:
        modelo, embedder = pickle.load(f)
    return modelo, embedder


def predecir(modelo, embedder, texto):
    texto = normalizar(texto)
    emb = embedder.encode([texto])

    pred = modelo.predict(emb)[0]
    probs = modelo.predict_proba(emb)[0]

    clases = modelo.classes_

    return {
        "etiqueta": pred,
        "confianza": dict(zip(clases, [round(p * 100, 1) for p in probs]))
    }


def main():
    modelo_path = os.path.join(os.path.dirname(__file__), "modelo.pkl")

    if not os.path.exists(modelo_path):
        print("No se encontró modelo.pkl")
        return

    modelo, embedder = cargar_modelo(modelo_path)

    print("=" * 50)
    print("Clasificador con embeddings")
    print("=" * 50)

    while True:
        texto = input("\n> ").strip()

        if texto.lower() in ["salir", "exit"]:
            break

        r = predecir(modelo, embedder, texto)

        icono = "💸" if r["etiqueta"] == "gasto" else "💰"

        print(f"\n{icono} {r['etiqueta'].upper()}")
        print(f"Gasto: {r['confianza'].get('gasto', 0):.1f}%")
        print(f"Ingreso: {r['confianza'].get('ingreso', 0):.1f}%")


if __name__ == "__main__":
    main()