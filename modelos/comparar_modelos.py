"""
Comparador de modelos: claude vs chat vs visualchat
====================================================
Evalúa los tres modelos sobre el mismo conjunto de 50 frases
y determina cuál es el mejor según precisión y confianza media.

Uso:
    python comparar_modelos.py

Debe ejecutarse desde la carpeta que contiene las subcarpetas
modeloclaude/, modelochatgptv2/ y modelochatvisual/.
"""

import os
import sys
import pickle
import re
import importlib.util
import random

# ── Frases de prueba comunes (las mismas 50 del evaluar.py compartido) ──────
FRASES_PRUEBA = [
    # GASTOS (25)
    ("me han pasado el recibo del gas de noventa euros",                       "gasto"),
    ("acabo de pagar doscientos cincuenta euros al mecánico",                  "gasto"),
    ("se me fueron ciento veinte euros en el supermercado esta semana",        "gasto"),
    ("le mandé cuarenta euros a mi hermana para la compra",                    "gasto"),
    ("pagué setenta y cinco euros de parking en el aeropuerto",                "gasto"),
    ("me han cobrado doce euros de comisión por la transferencia",             "gasto"),
    ("gasté trescientos euros en una chaqueta de invierno",                    "gasto"),
    ("tuve que abonar mil doscientos euros de matrícula universitaria",        "gasto"),
    ("me salió por cuarenta y cinco euros la revisión de la vista",            "gasto"),
    ("le hice una transferencia de cien euros a pedro para la cena",           "gasto"),
    ("me han cargado siete euros de youtube premium este mes",                 "gasto"),
    ("solté ochocientos euros en billetes de avión para las vacaciones",       "gasto"),
    ("me costaron sesenta euros las zapatillas en las rebajas",                "gasto"),
    ("pagamos doscientos euros entre todos para el regalo de cumpleaños",      "gasto"),
    ("me han cobrado treinta y cinco euros de mantenimiento de la cuenta",     "gasto"),
    ("acabé pagando cuatrocientos euros al fontanero por la avería",           "gasto"),
    ("se fueron noventa y nueve euros en la suscripción anual de amazon",      "gasto"),
    ("me tocó pagar cincuenta euros de multa por aparcar mal",                 "gasto"),
    ("desembolsé dos mil euros en un sofá nuevo para el salón",                "gasto"),
    ("me han pasado el recibo de la comunidad de ciento ochenta euros",        "gasto"),
    ("gastamos quinientos euros en la cena de navidad de la empresa",          "gasto"),
    ("me cobró treinta euros el electricista por cambiar el enchufe",          "gasto"),
    ("pagué mil quinientos euros de seguro del hogar este año",                "gasto"),
    ("se me fueron veinte euros en el taxi de vuelta a casa",                  "gasto"),
    ("me han cargado cuatrocientos euros de la cuota trimestral del gimnasio", "gasto"),
    # INGRESOS (25)
    ("me han ingresado mil ochocientos euros de la nómina de este mes",        "ingreso"),
    ("cobré trescientos cincuenta euros por un trabajo de diseño gráfico",     "ingreso"),
    ("me llegó un bizum de ochenta euros de laura por la cena",                "ingreso"),
    ("recibí dos mil quinientos euros de la herencia de mi abuelo",            "ingreso"),
    ("me han abonado noventa euros de cashback de la tarjeta",                 "ingreso"),
    ("cobré cuatrocientos euros dando clases particulares este mes",           "ingreso"),
    ("me devolvieron ciento veinte euros de hacienda esta semana",             "ingreso"),
    ("ingresé seiscientos euros por vender la bici en wallapop",               "ingreso"),
    ("me pagaron mil cien euros por el reportaje fotográfico",                 "ingreso"),
    ("he cobrado cincuenta euros vendiendo ropa en vinted",                    "ingreso"),
    ("me han transferido cuatrocientos cincuenta euros por el artículo",       "ingreso"),
    ("recibí setecientos euros de paga extraordinaria de verano",              "ingreso"),
    ("me cayeron tres mil euros de un premio de la lotería",                   "ingreso"),
    ("cobré doscientos diez euros por cuidar al perro de mis vecinos",         "ingreso"),
    ("me han abonado mil euros de la prestación por desempleo",                "ingreso"),
    ("recibí ciento cincuenta euros de regalo de mis padres por mi cumpleaños","ingreso"),
    ("me ingresaron ochocientos cincuenta euros de la beca del máster",        "ingreso"),
    ("cobré dos mil euros de honorarios como autónomo este trimestre",         "ingreso"),
    ("me han pagado trescientos euros por traducir un contrato",               "ingreso"),
    ("recibí quinientos euros de la devolución del seguro del coche",          "ingreso"),
    ("me llegó la transferencia de novecientos euros del alquiler",            "ingreso"),
    ("he cobrado setenta y cinco euros por vender el móvil antiguo",           "ingreso"),
    ("me han abonado doscientos veinte euros de comisión por las ventas",      "ingreso"),
    ("cobré mil seiscientos euros de nómina la semana pasada",                 "ingreso"),
    ("me han pagado sesenta euros por participar en un estudio de mercado",    "ingreso"),
]

VERBOS_GASTO = [
    "pague", "he pagado", "me cobraron", "gaste", "se me fueron",
    "me quitaron", "me sacaron", "solte", "desembolse"
]

VERBOS_INGRESO = [
    "me pagaron", "he cobrado", "me ingresaron", "recibi",
    "me dieron", "me entraron", "me abonaron", "me cayeron"
]

OBJETOS_GASTO = [
    "en comida", "en gasolina", "en ropa", "en el supermercado",
    "en el alquiler", "en una cena", "en suscripciones",
    "en el gimnasio", "en un vuelo", "en el coche"
]

OBJETOS_INGRESO = [
    "por un trabajo", "por una venta", "de la nomina",
    "de un proyecto", "de una transferencia",
    "de una devolucion", "de una ayuda",
    "de un premio", "de un bizum"
]

CANTIDADES = [
    "diez", "veinte", "treinta", "cuarenta", "cincuenta",
    "sesenta", "setenta", "ochenta", "noventa", "cien",
    "ciento veinte", "ciento cincuenta", "doscientos",
    "trescientos", "cuatrocientos", "quinientos",
    "mil", "mil doscientos", "mil quinientos", "dos mil"
]

MULETILLAS = [
    "", "esta mañana", "ayer", "hoy", "hace poco",
    "el otro dia", "recientemente", "sin darme cuenta"
]


def generar_frase_gasto():
    return f"{random.choice(VERBOS_GASTO)} {random.choice(CANTIDADES)} euros {random.choice(OBJETOS_GASTO)} {random.choice(MULETILLAS)}".strip()


def generar_frase_ingreso():
    return f"{random.choice(VERBOS_INGRESO)} {random.choice(CANTIDADES)} euros {random.choice(OBJETOS_INGRESO)} {random.choice(MULETILLAS)}".strip()


def generar_frases_test(n=500):
    frases = []
    for _ in range(n):
        if random.random() < 0.5:
            frases.append((generar_frase_gasto(), "gasto"))
        else:
            frases.append((generar_frase_ingreso(), "ingreso"))
    return frases

FRASES_PRUEBA += generar_frases_test(500)

# ── Carga dinámica del normalizar() y modelo de cada carpeta ────────────────

def cargar_normalizador(carpeta: str):
    """Importa la función normalizar() del clasificador.py de cada modelo."""
    clf_path = os.path.join(carpeta, "clasificador.py")
    spec = importlib.util.spec_from_file_location("clf_mod", clf_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.normalizar


def cargar_modelo_pkl(carpeta: str):
    pkl_path = os.path.join(carpeta, "modelo.pkl")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def evaluar_modelo(nombre: str, carpeta: str) -> dict:
    """Evalúa un modelo sobre las FRASES_PRUEBA y devuelve sus métricas."""
    normalizar = cargar_normalizador(carpeta)
    modelo     = cargar_modelo_pkl(carpeta)

    correctos       = 0
    confianzas_ok   = []
    confianzas_fail = []
    fallos          = []

    for frase, esperado in FRASES_PRUEBA:
        norm     = normalizar(frase)
        predicho = modelo.predict([norm])[0]
        probs    = modelo.predict_proba([norm])[0]
        clases   = list(modelo.classes_)
        confianza = probs[clases.index(predicho)] * 100

        ok = predicho == esperado
        if ok:
            correctos += 1
            confianzas_ok.append(confianza)
        else:
            confianzas_fail.append(confianza)
            fallos.append((frase, esperado, predicho, confianza))

    total    = len(FRASES_PRUEBA)
    precision = correctos / total * 100
    conf_ok   = sum(confianzas_ok)   / len(confianzas_ok)   if confianzas_ok   else 0.0
    conf_fail = sum(confianzas_fail) / len(confianzas_fail) if confianzas_fail else 0.0

    return {
        "nombre":    nombre,
        "total":     total,
        "correctos": correctos,
        "fallos":    fallos,
        "precision": precision,
        "conf_ok":   conf_ok,
        "conf_fail": conf_fail,
    }


# ── Puntuación combinada para determinar el ganador ─────────────────────────
def puntuacion(r: dict) -> float:
    """
    Combina precisión (80%) y confianza media en aciertos (20%).
    La precisión es el criterio principal; la confianza desempata.
    """
    return r["precision"] * 0.80 + r["conf_ok"] * 0.20


# ── Impresión de resultados detallados ──────────────────────────────────────

def imprimir_resultado(r: dict, pos: int):
    medalla = {1: "🥇", 2: "🥈", 3: "🥉"}.get(pos, "  ")
    print(f"\n{medalla}  MODELO: {r['nombre'].upper()}")
    print(f"   {'─' * 40}")
    print(f"   Frases evaluadas  : {r['total']}")
    print(f"   Aciertos          : {r['correctos']} / {r['total']}")
    print(f"   Precisión         : {r['precision']:.1f}%")
    print(f"   Confianza (✓)     : {r['conf_ok']:.1f}%")
    if r["fallos"]:
        print(f"   Confianza (✗)     : {r['conf_fail']:.1f}%")
        print(f"   Fallos ({len(r['fallos'])}):")
        for frase, esp, pred, conf in r["fallos"]:
            print(f"     · esp={esp:<10} pred={pred:<10} conf={conf:.1f}%")
            print(f"       \"{frase}\"")
    else:
        print(f"   Fallos            : ninguno ✓")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    base = os.path.dirname(os.path.abspath(__file__))

    modelos = [
        ("claude",      os.path.join(base, "modeloclaude")),
        ("chat",        os.path.join(base, "modelochatgptv2")),
        ("visualchat",  os.path.join(base, "modelochatvisual")),
    ]

    print("=" * 60)
    print("  COMPARADOR DE MODELOS — gasto vs ingreso")
    print(f"  Frases de prueba: {len(FRASES_PRUEBA)}")
    print("=" * 60)

    resultados = []
    for nombre, carpeta in modelos:
        if not os.path.isdir(carpeta):
            print(f"\n⚠️  Carpeta no encontrada: {carpeta}")
            continue
        try:
            print(f"\n  Evaluando {nombre}...", end=" ", flush=True)
            r = evaluar_modelo(nombre, carpeta)
            resultados.append(r)
            print(f"✓  ({r['precision']:.1f}%  conf {r['conf_ok']:.1f}%)")
        except Exception as e:
            print(f"\n  ✗ Error en {nombre}: {e}")

    if not resultados:
        print("\nNo se pudo evaluar ningún modelo.")
        return

    # Ordenar por puntuación combinada
    resultados.sort(key=puntuacion, reverse=True)

    print("\n" + "=" * 60)
    print("  RESULTADOS DETALLADOS (mejor → peor)")
    print("=" * 60)
    for pos, r in enumerate(resultados, 1):
        imprimir_resultado(r, pos)

    # ── Tabla resumen ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESUMEN COMPARATIVO")
    print("=" * 60)
    print(f"  {'Modelo':<14} {'Precisión':>10} {'Conf (✓)':>10} {'Fallos':>8}  {'Puntuación':>12}")
    print(f"  {'─'*14} {'─'*10} {'─'*10} {'─'*8}  {'─'*12}")
    for pos, r in enumerate(resultados, 1):
        medalla = {1: "🥇", 2: "🥈", 3: "🥉"}.get(pos, "  ")
        score = puntuacion(r)
        print(f"  {medalla} {r['nombre']:<12} {r['precision']:>9.1f}%"
              f" {r['conf_ok']:>9.1f}%"
              f" {len(r['fallos']):>8}"
              f"  {score:>11.2f}")

    # ── Ganador ──────────────────────────────────────────────────────
    ganador = resultados[0]
    print("\n" + "=" * 60)
    print(f"  🏆  MEJOR MODELO: {ganador['nombre'].upper()}")
    print(f"      Precisión {ganador['precision']:.1f}%  ·  "
          f"Confianza media {ganador['conf_ok']:.1f}%  ·  "
          f"Fallos: {len(ganador['fallos'])}")
    print("=" * 60)


if __name__ == "__main__":
    main()
