import pandas as pd
import random

class TitleGenerator:
    def __init__(self):
        # Estructuras gramaticales por categoría
        self.templates = {
            'Gaming': {
                'prefix': ['Speedrun', 'Glitches de', 'Gameplay 4K:', 'Final de', 'Cómo vencer a', 'Secretos de'],
                'subject': ['Elden Ring', 'Minecraft', 'Zelda', 'Fortnite', 'Roblox', 'GTA VI'],
                'suffix': ['en 5 minutos', 'sin morir', '(Increíble)', '2026 Update', 'Nivel Pro', '¡ÉPICO!']
            },
            'Education': {
                'prefix': ['Aprende', 'Guía de', 'La verdad sobre', 'Introducción a', 'Curso de', 'Secretos de'],
                'subject': ['Python', 'Física Cuántica', 'IA Generativa', 'Historia', 'Economía', 'Matemáticas'],
                'suffix': ['para principiantes', 'desde cero', 'en 10 minutos', 'Masterclass', '2026', 'fácil']
            },
            'Sports': {
                'prefix': ['Resumen:', 'Mejores momentos de', 'Escándalo en', 'Entrenamiento de', 'Análisis:'],
                'subject': ['el Clásico', 'la NBA', 'Wimbledon', 'la F1', 'Champions League', 'UFC'],
                'suffix': ['Highlights', '¡No lo creerás!', 'Temporada 2026', 'Final de infarto', 'HD']
            },
            'News': {
                'prefix': ['Última hora:', 'Reportaje sobre', 'Resumen de', 'Análisis de', 'Alerta:'],
                'subject': ['Política Global', 'Cambio Climático', 'Nuevas Tecnologías', 'Economía', 'Conflictos'],
                'suffix': ['Hoy', '2026', 'Urgente', 'Internacional', 'Exclusivo']
            },
            'Lifestyle': {
                'prefix': ['Mi rutina de', 'Vlog:', 'Cómo organizar', 'Recetas de', 'Día en'],
                'subject': ['Mañana', 'Viaje a Japón', 'Casa Nueva', 'Cocina Saludable', 'Productividad'],
                'suffix': ['2026', 'Realista', 'Tips', 'conmigo', 'Transformación']
            }
        }

    def generate(self, category):
            parts = self.templates.get(category, {
                'prefix': ['Video de'], 'subject': [category], 'suffix': ['']
            })
            t = f"{random.choice(parts['prefix'])} {random.choice(parts['subject'])} {random.choice(parts['suffix'])}"
            return t.strip()
