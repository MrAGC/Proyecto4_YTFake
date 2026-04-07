import random

class GeneradorMarkov:
    def __init__(self, orden=2):
        self.orden = orden
        self.memoria = {}
        self.token_inicio = "<START>"
        self.token_fin = "<END>"

    def _tokenizar(self, titulo):
        return str(titulo).strip().split()

    def entrenar(self, titulos):
        for titulo in titulos:
            palabras = self._tokenizar(titulo)
            if len(palabras) < 2:
                continue
            secuencia = ([self.token_inicio] * self.orden) + palabras + [self.token_fin]
            for i in range(len(secuencia) - self.orden):
                estado = tuple(secuencia[i:i + self.orden])
                siguiente = secuencia[i + self.orden]
                if estado not in self.memoria:
                    self.memoria[estado] = []
                self.memoria[estado].append(siguiente)

    def generar(self, inicio_palabra=None, max_palabras=15, min_palabras=4):
        if not self.memoria:
            return ""

        estado_inicial = tuple([self.token_inicio] * self.orden)
        if inicio_palabra:
            candidatos = self.memoria.get(estado_inicial, [])
            candidatos_filtrados = [
                palabra for palabra in candidatos
                if palabra.lower() == inicio_palabra.lower()
            ]
            if candidatos_filtrados:
                primera_palabra = random.choice(candidatos_filtrados)
                resultado = [primera_palabra]
                estado_actual = tuple(([self.token_inicio] * (self.orden - 1)) + [primera_palabra])
            else:
                estado_actual = estado_inicial
                resultado = []
        else:
            estado_actual = estado_inicial
            resultado = []

        for _ in range(max_palabras):
            opciones = self.memoria.get(estado_actual)
            if not opciones:
                break
            proxima = random.choice(opciones)
            if proxima == self.token_fin:
                if len(resultado) >= min_palabras:
                    break
                continue
            resultado.append(proxima)
            estado_actual = tuple((list(estado_actual) + [proxima])[-self.orden:])

        return " ".join(resultado)

