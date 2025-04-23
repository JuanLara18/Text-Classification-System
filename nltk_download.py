#!/usr/bin/env python3
"""
Script para descargar todos los recursos NLTK necesarios para el proyecto.
"""

import nltk
import os
import sys

def download_nltk_resources():
    """Descarga los recursos NLTK necesarios para el proyecto."""
    resources = ['punkt', 'stopwords', 'wordnet']
    
    print("Descargando recursos NLTK...")
    for resource in resources:
        try:
            print(f"Descargando {resource}...")
            nltk.download(resource)
            print(f"✅ {resource} descargado correctamente")
        except Exception as e:
            print(f"❌ Error descargando {resource}: {str(e)}")
    
    print("\nVerificando instalación:")
    for resource in resources:
        try:
            if resource == 'punkt':
                nltk.data.find('tokenizers/punkt')
            elif resource == 'stopwords':
                nltk.data.find('corpora/stopwords')
            elif resource == 'wordnet':
                nltk.data.find('corpora/wordnet')
            print(f"✅ {resource} verificado correctamente")
        except LookupError:
            print(f"❌ {resource} no está disponible")
    
    print("\nDirectorios de datos NLTK:")
    print(nltk.data.path)

if __name__ == "__main__":
    download_nltk_resources()