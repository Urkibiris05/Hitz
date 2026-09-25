import re
from pathlib import Path

def limpiar_texto(texto):
    """
    Aplica las expresiones regulares para limpiar las alucinaciones de Whisper.
    """
    patron_alucinaciones = r"(?i)[¡¿]?(suscr[íi]bete al canal|gracias por ver el v[íi]deo)[!\.\?]*"
    
    # Reemplazar las frases
    texto_limpio = re.sub(patron_alucinaciones, "", texto)
    
    # Eliminar saltos de línea sobrantes
    texto_limpio = re.sub(r'\n\s*\n', '\n', texto_limpio)
    
    return texto_limpio.strip()

def limpiar_directorio(ruta_entrada):
    """
    Lee todos los archivos de un directorio, los limpia y los guarda en uno nuevo.
    """
    dir_entrada = Path(ruta_entrada)
    
    # Comprobar que el directorio de entrada existe
    if not dir_entrada.is_dir():
        print(f"Error: La carpeta '{dir_entrada}' no existe.")
        return

    # Crear el directorio de salida junto al original (ej: "transcripciones_limpias")
    dir_salida = dir_salida = dir_entrada / "limpias"
    dir_salida.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Carpeta origen: {dir_entrada}")
    print(f"📁 Carpeta destino: {dir_salida}\n")

    archivos_procesados = 0
    
    # Extensiones de archivo que queremos procesar (Whisper suele sacar estos formatos)
    extensiones_validas = ['.txt', '.srt', '.vtt']

    # Iterar sobre todos los archivos de la carpeta de entrada
    for archivo_entrada in dir_entrada.iterdir():
        if archivo_entrada.is_file() and archivo_entrada.suffix.lower() in extensiones_validas:
            try:
                # 1. Leer el archivo original (usando codificación UTF-8 para evitar problemas con las tildes)
                with open(archivo_entrada, 'r', encoding='utf-8') as f:
                    contenido = f.read()
                
                # 2. Limpiar el contenido
                contenido_limpio = limpiar_texto(contenido)
                
                # 3. Guardar en la nueva carpeta con el mismo nombre
                archivo_salida = dir_salida / archivo_entrada.name
                with open(archivo_salida, 'w', encoding='utf-8') as f:
                    f.write(contenido_limpio)
                    
                print(f"✔ Procesado: {archivo_entrada.name}")
                archivos_procesados += 1
                
            except Exception as e:
                print(f"✖ Error procesando '{archivo_entrada.name}': {e}")
                
    print(f"\n✨ ¡Proceso completado! Se han limpiado {archivos_procesados} archivos.")

# ==========================================
# CÓMO USARLO
# ==========================================

# Escribe aquí la ruta de la carpeta donde tienes los archivos sucios.
# Puedes usar rutas absolutas (ej. "C:/Usuarios/TuNombre/Documentos/transcripciones") 
# o relativas si el script está en la misma carpeta.

ruta_mis_archivos = "GRABACIONES ECOE/TRANSCRIPCIONES/MIERCOLES/ronda_1" 

limpiar_directorio(ruta_mis_archivos)