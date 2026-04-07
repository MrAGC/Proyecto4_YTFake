import hashlib
import math
import re

import pandas as pd

import config_title as config
from markov_title_model import GeneradorMarkov


CATEGORY_ANGLES = {
    "Lifestyle": [
        "mi experiencia real",
        "lo que aprendí",
        "sin filtros",
        "lo que cambiaría hoy",
        "la versión más honesta",
        "todo lo que pasó",
        "lo bueno y lo malo",
        "lo que nadie te cuenta",
        "lo que sí funcionó",
        "la historia completa",
        "cómo fue de verdad",
        "lo que más me sorprendió",
    ],
    "Comedy": [
        "un desastre total",
        "salió peor de lo esperado",
        "nadie estaba preparado",
        "quedó todo grabado",
        "más incómodo de lo normal",
        "fue peor en directo",
        "me arrepentí rápido",
        "casi sale bien",
        "sin sentido pero divertidísimo",
        "terminó fatal",
        "fue más raro de lo que parecía",
        "y todavía no lo supero",
    ],
    "Gaming": [
        "todo lo que debes saber",
        "mi opinión sincera",
        "lo que nadie te explica",
        "cómo mejorar más rápido",
        "lo que cambió de verdad",
        "después de muchas horas",
        "sin humo",
        "con lo más útil primero",
        "la parte que casi nadie aprovecha",
        "lo que haría si empezara hoy",
        "paso a paso",
        "sin perder tiempo",
    ],
    "News": [
        "lo que se sabe hasta ahora",
        "claves para entenderlo",
        "el contexto completo",
        "todo lo confirmado",
        "qué cambia ahora",
        "el resumen más claro",
        "sin ruido",
        "lo más importante",
        "lo que todavía no encaja",
        "la cronología bien explicada",
        "con datos reales",
        "qué significa realmente",
    ],
    "Music": [
        "mi opinión sincera",
        "lo que mejor funcionó",
        "cómo suena de verdad",
        "el cambio que más se nota",
        "mi proceso real",
        "lo que aprendí haciéndolo",
        "sin gastar de más",
        "paso a paso",
        "con lo esencial",
        "lo que repetiría hoy",
        "lo bueno y lo mejorable",
        "sin vender humo",
    ],
    "Education": [
        "explicado fácil",
        "lo esencial primero",
        "sin memorizar de más",
        "lo que sí funciona",
        "con un método claro",
        "paso a paso",
        "sin complicarlo",
        "la forma más simple",
        "cómo entenderlo de verdad",
        "lo que debes dominar",
        "con ejemplos útiles",
        "sin agobiarte",
    ],
    "Sports": [
        "lo que más noté",
        "sin lesionarte",
        "mi experiencia real",
        "cómo hacerlo bien",
        "lo básico que sí importa",
        "lo que cambiaría hoy",
        "paso a paso",
        "sin complicarte",
        "lo que mejor me funcionó",
        "la parte que más cuesta",
        "lo que da resultado",
        "sin perder constancia",
    ],
    "Tech": [
        "mi opinión sincera",
        "lo bueno y lo malo",
        "lo que sí merece la pena",
        "lo más útil de verdad",
        "sin humo",
        "después de usarlo",
        "con lo importante primero",
        "lo que revisaría antes de comprar",
        "lo que cambió mi flujo de trabajo",
        "guía clara para empezar",
        "sin gastar de más",
        "la versión más práctica",
    ],
    "General": [
        "la historia completa",
        "lo que aprendí después",
        "sin filtros",
        "lo que nunca conté",
        "cómo fue de verdad",
        "todo el contexto",
        "lo que cambiaría hoy",
        "por qué cambió todo",
        "la versión más honesta",
        "lo que nadie sabía",
        "lo que vino después",
        "sin dramatizarlo",
    ],
}

OPENING_PHRASES = [
    "lo que de verdad pasó",
    "la parte que casi nadie cuenta",
    "la versión más honesta",
    "el contexto completo",
    "mi experiencia real",
    "lo que aprendí",
    "lo más importante",
    "sin filtros",
    "la historia bien contada",
    "lo que más me sorprendió",
    "lo esencial",
    "la explicación clara",
]

SHORT_RUNTIME_TAGS = [
    "versión corta",
    "al grano",
    "en pocos minutos",
    "sin rodeos",
]

MID_RUNTIME_TAGS = [
    "con buen contexto",
    "bien explicado",
    "con ejemplos",
    "paso a paso",
]

LONG_RUNTIME_TAGS = [
    "la versión completa",
    "con más detalle",
    "sin dejar fuera lo importante",
    "la explicación completa",
]

HIGH_PERFORMANCE_TAGS = [
    "más interesante de lo que parecía",
    "me sorprendió bastante",
    "de lo mejor que he probado",
    "funcionó mejor de lo esperado",
    "dio mucho más de sí",
    "enganchó desde el principio",
]

MID_PERFORMANCE_TAGS = [
    "vale la pena entenderlo bien",
    "tiene más miga de la que parece",
    "merece una mirada más atenta",
    "acabó siendo mejor de lo esperado",
    "tiene detalles importantes",
    "da para explicarlo bien",
]

LOW_PERFORMANCE_TAGS = [
    "tuve sensaciones mixtas",
    "no fue exactamente lo que esperaba",
    "tuvo partes muy buenas y otras no tanto",
    "me dejó más dudas de las esperadas",
    "no era tan simple como parecía",
    "acabó siendo distinto de lo que imaginaba",
]

GENERIC_UNIQUENESS_TAGS = [
    "mi versión más completa",
    "con contexto real",
    "sin adornarlo",
    "explicado con calma",
    "en detalle",
    "bien resumido",
    "con lo esencial",
    "sin perder lo importante",
    "de principio a fin",
    "para entenderlo mejor",
    "con una mirada práctica",
    "más claro imposible",
]


class MotorTitulosCSV:
    def __init__(self):
        self.modelos = {}
        self.hooks_data = pd.DataFrame()
        self.training_titles_by_category = {}
        self.titulos_generados = set()
        self._cargar_y_entrenar()

    def _cargar_y_entrenar(self):
        try:
            df_train = pd.read_csv(config.TITLES_TRAINING_PATH)
            df_train["frase"] = df_train["frase"].fillna("").astype(str).str.strip()
            df_train = df_train[df_train["frase"] != ""]
            df_train = df_train[df_train["frase"].apply(self._es_frase_entrenable)]

            for cat, group in df_train.groupby(config.CATEGORY_COLUMN):
                frases = group["frase"].tolist()
                gm = GeneradorMarkov(orden=config.MARKOV_ORDER)
                gm.entrenar(frases)
                self.modelos[cat] = gm
                self.training_titles_by_category[cat] = frases

            self.hooks_data = pd.read_csv(config.HOOKS_CSV_PATH)
            self.hooks_data["hook"] = self.hooks_data["hook"].fillna("").astype(str).str.strip()
            self.hooks_data = self.hooks_data[self.hooks_data["hook"] != ""]

            print(f"Modelos entrenados: {list(self.modelos.keys())}")
            print(f"Hooks cargados: {len(self.hooks_data)} registros.")
        except Exception as e:
            print(f"Error en carga o entrenamiento: {e}")

    def _es_frase_entrenable(self, texto):
        texto = self._normalizar_espacios(texto)
        return (
            len(texto) >= config.MIN_TITLE_LENGTH
            and len(texto.split()) >= config.MIN_WORDS_TITLE
        )

    def _normalizar_espacios(self, texto):
        return re.sub(r"\s+", " ", str(texto or "")).strip()

    def _capitalizar_titulo(self, texto):
        texto = self._normalizar_espacios(texto)
        if not texto:
            return ""
        for indice, caracter in enumerate(texto):
            if caracter.isalpha():
                return texto[:indice] + caracter.upper() + texto[indice + 1:]
        return texto

    def _normalizar_puntuacion(self, texto):
        texto = self._normalizar_espacios(texto)
        reemplazos = {
            " ,": ",",
            " .": ".",
            " :": ":",
            " ;": ";",
            " !": "!",
            " ?": "?",
            "..": ".",
            "!!": "!",
            "??": "?",
            "Â¿ ": "¿",
            "Â¡ ": "¡",
            "¿ ": "¿",
            "¡ ": "¡",
            " : ": ": ",
        }
        for viejo, nuevo in reemplazos.items():
            texto = texto.replace(viejo, nuevo)

        texto = re.sub(r"\s+([,.;:!?])", r"\1", texto)
        texto = re.sub(r"([¿¡])\s+", r"\1", texto)
        texto = re.sub(r"\s+\)", ")", texto)
        texto = re.sub(r"\(\s+", "(", texto)
        texto = re.sub(r"\s+\]", "]", texto)
        texto = re.sub(r"\[\s+", "[", texto)
        texto = texto.strip(" -|:.")
        return self._capitalizar_titulo(texto)

    def _recortar_final(self, texto):
        palabras = self._normalizar_espacios(texto).split()
        while palabras and palabras[-1].lower().strip(",.:;!?") in config.STOPWORDS_FINALES:
            palabras.pop()
        return " ".join(palabras)

    def _es_titulo_usable(self, texto):
        texto = self._normalizar_espacios(texto)
        if not texto:
            return False
        if len(texto) < config.MIN_TITLE_LENGTH or len(texto) > config.MAX_TITLE_LENGTH:
            return False
        if len(texto.split()) < config.MIN_WORDS_TITLE:
            return False
        if any(patron in texto for patron in config.PATRONES_TITULO_INVALIDO):
            return False
        if texto[-1] in {":", "-", "|", "/", ","}:
            return False
        if sum(1 for c in texto if c in "!?") > 2:
            return False
        if re.search(r"\b(\w+)( \1){2,}\b", texto, flags=re.IGNORECASE):
            return False
        return True

    def _titulo_demasiado_parecido(self, texto):
        return texto.lower() in self.titulos_generados

    def _fila_key(self, row, salt):
        valores = [
            str(row.get("video_id", "")),
            str(row.get(config.CATEGORY_COLUMN, "")),
            str(row.get(config.DURATION_COLUMN, "")),
            str(row.get("total_views", "")),
            str(row.get("total_likes", "")),
            str(row.get("total_comments", "")),
            str(row.get("avg_watch_percent", "")),
            str(row.get("click_through_rate", "")),
            salt,
        ]
        return hashlib.sha256("|".join(valores).encode("utf-8")).hexdigest()

    def _stable_index(self, row, salt, modulo):
        if modulo <= 0:
            return 0
        return int(self._fila_key(row, salt)[:12], 16) % modulo

    def _stable_pick(self, opciones, row, salt):
        if not opciones:
            return ""
        return opciones[self._stable_index(row, salt, len(opciones))]

    def _safe_float(self, valor):
        try:
            numero = float(valor)
        except (TypeError, ValueError):
            return 0.0
        return numero if math.isfinite(numero) else 0.0

    def _fallback_categoria(self, categoria):
        opciones = self.training_titles_by_category.get(categoria) or []
        if opciones:
            return opciones[0]
        return f"{categoria} en 2026"

    def _generar_base(self, categoria, row):
        gm = self.modelos.get(categoria)
        opciones_reales = self.training_titles_by_category.get(categoria) or []

        if gm is None:
            return self._fallback_categoria(categoria)

        if opciones_reales and self._stable_index(row, "raw-title", 100) < 20:
            base = self._stable_pick(opciones_reales, row, "raw-title-choice")
            return self._recortar_final(self._normalizar_puntuacion(base))

        for intento in range(8):
            base = gm.generar(
                max_palabras=config.MAX_WORDS_TITLE,
                min_palabras=config.MIN_WORDS_TITLE,
            )
            base = self._recortar_final(self._normalizar_puntuacion(base))
            if self._es_titulo_usable(base):
                return base

        return self._fallback_categoria(categoria)

    def _runtime_minutes(self, row):
        segundos = max(0, int(round(self._safe_float(row.get(config.DURATION_COLUMN, 0)))))
        minutos = max(1, int(round(segundos / 60)))
        return segundos, minutos

    def _performance_pool(self, row):
        like_rate = self._safe_float(row.get("like_rate", 0))
        comment_rate = self._safe_float(row.get("comment_rate", 0))
        watch = self._safe_float(row.get("avg_watch_percent", 0))
        ctr = self._safe_float(row.get("click_through_rate", 0))
        score = (like_rate * 0.35) + (comment_rate * 0.15) + (watch * 0.25) + (min(ctr, 2.5) / 2.5 * 0.25)
        if score >= 0.72:
            return HIGH_PERFORMANCE_TAGS
        if score >= 0.48:
            return MID_PERFORMANCE_TAGS
        return LOW_PERFORMANCE_TAGS

    def _runtime_pool(self, row):
        segundos, minutos = self._runtime_minutes(row)
        if segundos <= 240:
            pool = SHORT_RUNTIME_TAGS
        elif segundos <= 1500:
            pool = MID_RUNTIME_TAGS
        else:
            pool = LONG_RUNTIME_TAGS
        return pool, minutos

    def _elegir_hook(self, row):
        if self.hooks_data.empty:
            return ""

        duracion_val = self._safe_float(row.get(config.DURATION_COLUMN, 0))
        if duracion_val < config.MIN_HOOK_DURATION:
            return ""

        prob_hook = 18 + self._stable_index(row, "hook-prob", 18)
        if self._stable_index(row, "hook-trigger", 100) >= prob_hook:
            return ""

        valid_hooks = self.hooks_data[
            (self.hooks_data["min_duracion"] <= duracion_val)
            & (self.hooks_data["max_duracion"] >= duracion_val)
        ]
        if valid_hooks.empty:
            return ""

        categoria = str(row.get(config.CATEGORY_COLUMN, ""))
        clickbait_suave = valid_hooks[valid_hooks["nivel_clickbait"] != "alto"]
        if categoria in {"Education", "Tech", "News", "Sports"} and not clickbait_suave.empty:
            valid_hooks = clickbait_suave

        hooks = sorted(valid_hooks["hook"].astype(str).str.strip().unique().tolist())
        return self._stable_pick(hooks, row, "hook-choice")

    def _contexto_fila(self, row):
        categoria = str(row.get(config.CATEGORY_COLUMN, "")).strip()
        runtime_pool, minutos = self._runtime_pool(row)
        performance_pool = self._performance_pool(row)
        angle_pool = CATEGORY_ANGLES.get(categoria, CATEGORY_ANGLES["General"])

        return {
            "angle": self._stable_pick(angle_pool, row, "angle"),
            "angle_alt": self._stable_pick(angle_pool, row, "angle-alt"),
            "opening": self._stable_pick(OPENING_PHRASES, row, "opening"),
            "runtime_tag": self._stable_pick(runtime_pool, row, "runtime"),
            "performance_tag": self._stable_pick(performance_pool, row, "performance"),
            "unique_tag": self._stable_pick(GENERIC_UNIQUENESS_TAGS, row, "unique-tag"),
            "minutes": minutos,
        }

    def _candidatos_titulo(self, base, tema, hook, row):
        contexto = self._contexto_fila(row)
        base = self._recortar_final(base)
        tema = self._normalizar_espacios(tema)
        hook = self._normalizar_espacios(hook)

        candidatos = []
        usar_tema = bool(tema)
        usar_hook = bool(hook)

        if usar_tema:
            candidatos.extend([
                f"{base}: {tema}",
                f"{tema}: {base}",
                f"{base} | {tema}",
                f"{base}: {contexto['angle']} en {tema}",
                f"{contexto['opening']}: {base} en {tema}",
                f"{base} | {contexto['runtime_tag']} en {tema}",
                f"{base}: {contexto['performance_tag']} en {tema}",
            ])
            if usar_hook:
                candidatos.extend([
                    f"{hook}: {base} en {tema}",
                    f"{base} | {hook} | {tema}",
                    f"{hook}: {base} | {contexto['angle']}",
                ])
        else:
            candidatos.extend([
                base,
                f"{base}: {contexto['angle']}",
                f"{contexto['opening']}: {base}",
                f"{base} | {contexto['performance_tag']}",
                f"{base}: {contexto['runtime_tag']}",
                f"{base} ({contexto['unique_tag']})",
                f"{base}: {contexto['angle']} y {contexto['performance_tag']}",
                f"{base} | {contexto['runtime_tag']} | {contexto['unique_tag']}",
                f"{base}: {contexto['opening']}",
            ])
            if usar_hook:
                candidatos.extend([
                    f"{hook}: {base}",
                    f"{base} | {hook}",
                    f"{hook}: {base} | {contexto['angle']}",
                    f"{hook} | {base} | {contexto['runtime_tag']}",
                ])

        return [self._normalizar_puntuacion(c) for c in candidatos]

    def _variaciones_unicas(self, titulo, row):
        contexto = self._contexto_fila(row)
        minutos = contexto["minutes"]

        variantes = [
            f"{titulo}: {contexto['unique_tag']}",
            f"{titulo}: {contexto['performance_tag']}",
            f"{titulo} | {contexto['runtime_tag']}",
            f"{titulo} | {contexto['angle_alt']}",
            f"{titulo} ({minutos} min)",
            f"{titulo}: {contexto['opening']}",
            f"{titulo} ({minutos} min, {contexto['runtime_tag']})",
            f"{titulo}: {contexto['angle_alt']} en {minutos} min",
            f"{titulo} | {contexto['performance_tag']} | {minutos} min",
            f"{titulo}: {contexto['unique_tag']} ({minutos} min)",
        ]

        return [self._normalizar_puntuacion(v) for v in variantes]

    def generar(self, row):
        categoria = str(row.get(config.CATEGORY_COLUMN, "")).strip()
        tema_str = str(row.get(config.TEMA_COLUMN, "")).strip() if pd.notna(row.get(config.TEMA_COLUMN, "")) else ""

        for intento in range(config.GENERATION_ATTEMPTS):
            base = self._generar_base(categoria, row)
            hook = self._elegir_hook(row)
            candidatos = self._candidatos_titulo(base, tema_str, hook, row)

            desplazamiento = self._stable_index(row, f"candidate-offset-{intento}", max(1, len(candidatos)))
            candidatos = candidatos[desplazamiento:] + candidatos[:desplazamiento]

            for candidato in candidatos:
                if not self._es_titulo_usable(candidato):
                    continue
                if not self._titulo_demasiado_parecido(candidato):
                    self.titulos_generados.add(candidato.lower())
                    return candidato

                for variante in self._variaciones_unicas(candidato, row):
                    if self._es_titulo_usable(variante) and not self._titulo_demasiado_parecido(variante):
                        self.titulos_generados.add(variante.lower())
                        return variante

        fallback = self._normalizar_puntuacion(self._fallback_categoria(categoria))
        for variante in [fallback] + self._variaciones_unicas(fallback, row):
            if self._es_titulo_usable(variante) and not self._titulo_demasiado_parecido(variante):
                self.titulos_generados.add(variante.lower())
                return variante

        ultimo_recurso = self._normalizar_puntuacion(f"{fallback} ({self._runtime_minutes(row)[1]} min)")
        self.titulos_generados.add(ultimo_recurso.lower())
        return ultimo_recurso


def procesar():
    df_videos = pd.read_csv(config.VIDEOS_CSV_PATH)
    motor = MotorTitulosCSV()

    print("Generando títulos realistas...")
    df_videos[config.OUTPUT_TITLE_COLUMN] = df_videos.apply(
        motor.generar,
        axis=1,
    )

    df_videos.to_csv(config.OUTPUT_CSV_PATH, index=False)
    print(f"Proceso completado. Archivo creado en: {config.OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    procesar()
