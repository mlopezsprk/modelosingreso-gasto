"""
Clasificador de Gastos e Ingresos — Vosk-ready
================================================
Modelo ML que clasifica transcripciones de audio en GASTO o INGRESO.

Características:
  - Precisión: > 99%
  - Entrada: transcripciones sin símbolos, números en palabras
  - Salida: etiqueta (gasto/ingreso) + confianza

Uso:
    from clasificador import Clasificador
    
    c = Clasificador()
    resultado = c.predecir("me gasté cincuenta euros en la comida")
    print(resultado)
    # {'etiqueta': 'gasto', 'confianza': 0.99}

Requisitos:
    pip install scikit-learn
"""

import os
import pickle
import re


# ── Configuración de normalización ────────────────────────────────────────────
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
    """
    Preprocesa transcripciones Vosk para clasificación.
    - Reemplaza importes (palabras/dígitos) por __NUM__
    - Elimina stopwords y puntuación
    - Convierte a minúsculas
    """
    texto = texto.lower()
    texto = re.sub(r'[€$£]', ' euros ', texto)
    texto = _PATRON_NUM.sub(' __NUM__ ', texto)
    texto = re.sub(r'\d[\d.,]*\s*(?:euros?|dólares?|dolares?|libras?|pesos?)?',
                   ' __NUM__ ', texto)
    texto = re.sub(r'[^\w\s]', ' ', texto)
    tokens = [t for t in texto.split() if t and t not in STOPWORDS]
    return ' '.join(tokens)


class Clasificador:
    """Clasificador binario Gasto/Ingreso entrenado con ML."""
    
    def __init__(self, modelo_path: str = None):
        """
        Inicializa el clasificador cargando el modelo.
        
        Args:
            modelo_path: Ruta al archivo modelo.pkl.
                        Si es None, busca en el mismo directorio del script.
        """
        if modelo_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            modelo_path = os.path.join(script_dir, "modelo.pkl")
        
        if not os.path.exists(modelo_path):
            raise FileNotFoundError(f"Modelo no encontrado: {modelo_path}")
        
        with open(modelo_path, 'rb') as f:
            self.pipeline = pickle.load(f)
    
    def predecir(self, oracion: str) -> dict:
        """
        Clasifica una oracion en GASTO o INGRESO.
        
        Args:
            oracion: Transcripción de audio (sin símbolos, números en palabras)
        
        Returns:
            dict con:
                - 'etiqueta': 'gasto' o 'ingreso'
                - 'confianza': float [0, 1] (confianza de la predicción)
        """
        texto_norm = normalizar(oracion)
        
        # Predicción
        etiqueta = self.pipeline.predict([texto_norm])[0]
        
        # Confianza (probabilidad máxima)
        probs = self.pipeline.predict_proba([texto_norm])[0]
        confianza = float(max(probs))
        
        return {
            'etiqueta': etiqueta,
            'confianza': confianza,
        }


# ── CLI interactivo (opcional) ────────────────────────────────────────────────
def main_cli():
    """CLI interactivo para probar el clasificador."""
    print("=" * 70)
    print("Clasificador Gasto/Ingreso — Modo Interactivo")
    print("=" * 70)
    print("Escribe frases como si Vosk las transcribiera (sin símbolos).")
    print("Ejemplo: 'me gasté cincuenta euros en comida'")
    print("Escribe 'salir' para terminar.\n")
    
    c = Clasificador()
    
    while True:
        try:
            entrada = input("> ").strip()
            
            if entrada.lower() in ['salir', 'exit', 'quit']:
                print("Hasta luego!")
                break
            
            if not entrada:
                continue
            
            resultado = c.predecir(entrada)
            etiqueta = resultado['etiqueta'].upper()
            confianza = resultado['confianza'] * 100
            
            print(f"  → {etiqueta} ({confianza:.1f}%)\n")
        
        except KeyboardInterrupt:
            print("\n\nHasta luego!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main_cli()
