# --- CONFIGURACIÓN DE RUTAS ---
VIDEOS_CSV_PATH = "data/videos.csv"
OUTPUT_CSV_PATH = "data/videos_con_titulos.csv"

TITLES_TRAINING_PATH = "data/titulos.csv"
HOOKS_CSV_PATH = "data/hooks_emociones.csv"

# --- COLUMNAS DEL CSV ---
TEMA_COLUMN = "que_pasa"
CATEGORY_COLUMN = "category"
DURATION_COLUMN = "video_duration_s"
OUTPUT_TITLE_COLUMN = "titulo"

# --- PARÁMETROS DEL MODELO ---
MARKOV_ORDER = 3    # 1 para más creatividad/variedad, 3 para más coherencia
MAX_WORDS_TITLE = 20  # Largo máximo del título generado
CHANCE_OF_HOOK = 0.3  # 30% de probabilidad de añadir un emoji o gancho visual
MIN_WORDS_TITLE = 4
MIN_TITLE_LENGTH = 16
MAX_TITLE_LENGTH = 92
GENERATION_ATTEMPTS = 12
MIN_HOOK_DURATION = 45

STOPWORDS_FINALES = {
    "a", "al", "con", "de", "del", "el", "en", "la", "las",
    "los", "o", "para", "por", "que", "sin", "un", "una", "y",
}

PALABRAS_MINUSCULAS = {
    "a", "al", "con", "contra", "de", "del", "desde", "e", "el",
    "en", "entre", "hacia", "hasta", "la", "las", "lo", "los",
    "o", "para", "pero", "por", "que", "se", "sin", "su", "sus",
    "un", "una", "y",
}

PATRONES_TITULO_INVALIDO = [
    "[]", "{}", "()", "||", "::", "??", "!!", "  ",
]

# --- ESTRUCTURAS DINÁMICAS ---

# Categorizadas para facilitar la selección en el generador
# --- ESTRUCTURAS OPTIMIZADAS ---
ESTRUCTURAS_COMPLETAS = [
    "{hook}: {base} en {tema}",
    "{base}: lo que pasó en {tema}",
    "{tema}: {base}",
    "{base} | Mi experiencia en {tema}",
    "Lo que aprendí sobre {tema}: {base}",
    "{hook} | {base} en {tema}",
]

ESTRUCTURAS_SIN_HOOK = [
    "{base}: {tema}",
    "{tema} | {base}",
    "{base} en {tema}",
    "Lo que nadie te cuenta sobre {tema}",
    "Mi experiencia con {tema}: {base}",
    "Qué pasa con {tema}: {base}",
]

ESTRUCTURAS_SOLO_BASE_HOOK = [
    "{hook}: {base}",
    "{base} | {hook}",
    "{base}: {hook}",
    "{hook} | {base}",
]

ESTRUCTURAS_MINIMALISTAS = [
    "{base}",
    "{base}: lo que aprendí",
    "Mi experiencia con {base}",
    "Lo que nadie te cuenta de {base}",
    "{base} explicado fácil",
]
