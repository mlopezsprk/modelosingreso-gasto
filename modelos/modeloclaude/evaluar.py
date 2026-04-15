"""
Evaluación del clasificador con 50 frases de prueba
=====================================================
Frases escritas en estilo transcripción Vosk:
  - sin símbolos de moneda
  - importes en palabras
  - lenguaje coloquial y variado
  - NO presentes en el dataset de entrenamiento

Uso:
    python evaluar.py
"""

import os
import pickle
import re
import random

# ── Copiar normalizar() del clasificador principal ─────────────────────────
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
    texto = texto.lower()
    texto = re.sub(r'[€$£]', ' euros ', texto)
    texto = _PATRON_NUM.sub(' __NUM__ ', texto)
    texto = re.sub(r'\d[\d.,]*\s*(?:euros?|dólares?|dolares?|libras?|pesos?)?',
                   ' __NUM__ ', texto)
    texto = re.sub(r'[^\w\s]', ' ', texto)
    tokens = [t for t in texto.split() if t and t not in STOPWORDS]
    return ' '.join(tokens)


# ── 50 frases de prueba (estilo Vosk, fuera del dataset de entrenamiento) ──
FRASES_PRUEBA = [
    # ── GASTOS (25) ──────────────────────────────────────────────────
    ("me han pasado el recibo del gas de noventa euros",                          "gasto"),
    ("acabo de pagar doscientos cincuenta euros al mecánico",                     "gasto"),
    ("se me fueron ciento veinte euros en el supermercado esta semana",           "gasto"),
    ("le mandé cuarenta euros a mi hermana para la compra",                       "gasto"),
    ("pagué setenta y cinco euros de parking en el aeropuerto",                   "gasto"),
    ("me han cobrado doce euros de comisión por la transferencia",                "gasto"),
    ("gasté trescientos euros en una chaqueta de invierno",                       "gasto"),
    ("tuve que abonar mil doscientos euros de matrícula universitaria",           "gasto"),
    ("me salió por cuarenta y cinco euros la revisión de la vista",               "gasto"),
    ("le hice una transferencia de cien euros a pedro para la cena",              "gasto"),
    ("me han cargado siete euros de youtube premium este mes",                    "gasto"),
    ("solté ochocientos euros en billetes de avión para las vacaciones",          "gasto"),
    ("me costaron sesenta euros las zapatillas en las rebajas",                   "gasto"),
    ("pagamos doscientos euros entre todos para el regalo de cumpleaños",         "gasto"),
    ("me han cobrado treinta y cinco euros de mantenimiento de la cuenta",        "gasto"),
    ("acabé pagando cuatrocientos euros al fontanero por la avería",              "gasto"),
    ("se fueron noventa y nueve euros en la suscripción anual de amazon",         "gasto"),
    ("me tocó pagar cincuenta euros de multa por aparcar mal",                    "gasto"),
    ("desembolsé dos mil euros en un sofá nuevo para el salón",                   "gasto"),
    ("me han pasado el recibo de la comunidad de ciento ochenta euros",           "gasto"),
    ("gastamos quinientos euros en la cena de navidad de la empresa",             "gasto"),
    ("me cobró treinta euros el electricista por cambiar el enchufe",             "gasto"),
    ("pagué mil quinientos euros de seguro del hogar este año",                   "gasto"),
    ("se me fueron veinte euros en el taxi de vuelta a casa",                     "gasto"),
    ("me han cargado cuatrocientos euros de la cuota trimestral del gimnasio",    "gasto"),

    # ── INGRESOS (25) ────────────────────────────────────────────────
    ("me han ingresado mil ochocientos euros de la nómina de este mes",           "ingreso"),
    ("cobré trescientos cincuenta euros por un trabajo de diseño gráfico",        "ingreso"),
    ("me llegó un bizum de ochenta euros de laura por la cena",                   "ingreso"),
    ("recibí dos mil quinientos euros de la herencia de mi abuelo",               "ingreso"),
    ("me han abonado noventa euros de cashback de la tarjeta",                    "ingreso"),
    ("cobré cuatrocientos euros dando clases particulares este mes",              "ingreso"),
    ("me devolvieron ciento veinte euros de hacienda esta semana",                "ingreso"),
    ("ingresé seiscientos euros por vender la bici en wallapop",                  "ingreso"),
    ("me pagaron mil cien euros por el reportaje fotográfico",                    "ingreso"),
    ("he cobrado cincuenta euros vendiendo ropa en vinted",                       "ingreso"),
    ("me han transferido cuatrocientos cincuenta euros por el artículo",          "ingreso"),
    ("recibí setecientos euros de paga extraordinaria de verano",                 "ingreso"),
    ("me cayeron tres mil euros de un premio de la lotería",                      "ingreso"),
    ("cobré doscientos diez euros por cuidar al perro de mis vecinos",            "ingreso"),
    ("me han abonado mil euros de la prestación por desempleo",                   "ingreso"),
    ("recibí ciento cincuenta euros de regalo de mis padres por mi cumpleaños",   "ingreso"),
    ("me ingresaron ochocientos cincuenta euros de la beca del máster",           "ingreso"),
    ("cobré dos mil euros de honorarios como autónomo este trimestre",            "ingreso"),
    ("me han pagado trescientos euros por traducir un contrato",                  "ingreso"),
    ("recibí quinientos euros de la devolución del seguro del coche",             "ingreso"),
    ("me llegó la transferencia de novecientos euros del alquiler",               "ingreso"),
    ("he cobrado setenta y cinco euros por vender el móvil antiguo",              "ingreso"),
    ("me han abonado doscientos veinte euros de comisión por las ventas",         "ingreso"),
    ("cobré mil seiscientos euros de nómina la semana pasada",                    "ingreso"),
    ("me han pagado sesenta euros por participar en un estudio de mercado",       "ingreso"),
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


def main():
    modelo_path = os.path.join(os.path.dirname(__file__), 'modelo.pkl')
    if not os.path.exists(modelo_path):
        print(f"Error: no se encontró '{modelo_path}'.")
        return

    with open(modelo_path, 'rb') as f:
        modelo = pickle.load(f)

    print("=" * 65)
    print("  EVALUACIÓN DEL MODELO — 50 frases de prueba")
    print("=" * 65)
    print(f"\n{'#':<4} {'ESPERADO':<10} {'PREDICHO':<10} {'OK':<5} {'CONFIANZA':>10}  FRASE")
    print("─" * 120)

    correctos        = 0
    confianzas_ok    = []
    confianzas_fail  = []

    for i, (frase, esperado) in enumerate(FRASES_PRUEBA, 1):
        norm      = normalizar(frase)
        predicho  = modelo.predict([norm])[0]
        probs     = modelo.predict_proba([norm])[0]
        clases    = list(modelo.classes_)
        confianza = probs[clases.index(predicho)] * 100

        ok = predicho == esperado
        if ok:
            correctos += 1
            confianzas_ok.append(confianza)
        else:
            confianzas_fail.append(confianza)

        marca = "✓" if ok else "✗"
        color = "" if ok else "  ← FALLO"
        print(f"{i:<4} {esperado:<10} {predicho:<10} {marca:<5} {confianza:>8.1f}%  {frase[:70]}{color}")

    total     = len(FRASES_PRUEBA)
    precision = correctos / total * 100
    conf_media_ok   = sum(confianzas_ok)   / len(confianzas_ok)   if confianzas_ok   else 0
    conf_media_fail = sum(confianzas_fail) / len(confianzas_fail) if confianzas_fail else 0

    print("\n" + "=" * 65)
    print("  RESUMEN")
    print("=" * 65)
    print(f"  Frases evaluadas     : {total}")
    print(f"  Aciertos             : {correctos} / {total}")
    print(f"  Precisión            : {precision:.1f}%")
    print(f"  Confianza media (✓)  : {conf_media_ok:.1f}%")
    if confianzas_fail:
        print(f"  Confianza media (✗)  : {conf_media_fail:.1f}%")
    print("=" * 65)


if __name__ == "__main__":
    main()
