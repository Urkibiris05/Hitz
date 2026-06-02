import os
import glob
import whisper
import ollama
from jiwer import wer

# ==========================================
# 1. TRANSCRIPCIÓN CON WHISPER
# ==========================================
def transcribir_audio(ruta_audio, modelo_whisper="medium"):
    print(f"Cargando modelo Whisper ({modelo_whisper})...")
    modelo = whisper.load_model(modelo_whisper)
    
    print("Transcribiendo audio...")
    resultado = modelo.transcribe(ruta_audio, language="es")
    return resultado["text"]

# ==========================================
# 2. CÁLCULO DE ERROR (WER)
# ==========================================
def calcular_wer(transcripcion_manual, transcripcion_automatica):
    print("Calculando Tasa de Error de Palabras (WER)...")
    # JiWER calcula el ratio (0.0 a 1.0+), lo multiplicamos por 100 para el %
    tasa_error = wer(transcripcion_manual, transcripcion_automatica)
    return tasa_error * 100

# ==========================================
# 3. DIARIZACIÓN MEDIANTE OLLAMA
# ==========================================
def diarizar_con_ollama(texto_plano, modelo_llm="llama3"):
    print(f"Iniciando diarización con Ollama (Modelo: {modelo_llm})...")
    
    prompt_diarizacion = f"""
Te voy a mandar una conversación entre medico y paciente, y necesito que me digas quien de los dos es el que
esta diciendo cada linea de la conversación. Como output, quiero que me devuelvas la propia conversación pero
acotando al principio de cada linea si es el MEDICO: o el PACIENTE: el que la dice.
Ahi te va la conversación:

    {texto_plano}
    """
    
    respuesta = ollama.chat(model=modelo_llm, messages=[
        {'role': 'user', 'content': prompt_diarizacion}
    ])
    
    return respuesta['message']['content']

# ==========================================
# 4. AUTO-EVALUACIÓN MEDIANTE OLLAMA
# ==========================================
def evaluar_consulta_con_ollama(texto_estructurado, modelo_llm="gemma2"):
    print(f"Iniciando evaluación clínica con Ollama (Modelo: {modelo_llm})...")
    
    prompt = f"""Te voy a mandar unos criterios que sirven para evaluar como de correctamente sigue un medico su protocolo a la hora de consultar a un paciente. Junto con los criterios, te voy a mandar una conversacion entre un medico y un paciente.
Basandote en los criterios, quiero que me evalues la puntuación que consigue el medico en esta conversación.
Criterios:
{{
  "Comunicación": {{
    "Puntos totales": 5,
    "Subpuntos": {{
      "Se presenta al Paciente": {{
        "PuntosPregunta": 5
      }}
    }}
  }},
  "Identificación del Paciente (Datos de filiación)": {{
    "Puntos totales": 5,
    "Subpuntos": {{
      "Pregunta su nombre": {{
        "PuntosPregunta": 2
      }},
      "Edad": {{
        "PuntosPregunta": 1
      }},
      "Estado Civil": {{
        "PuntosPregunta": 1
      }},
      "Lugar de procedencia": {{
        "PuntosPregunta": 1
      }}
    }}
  }},
  "Motivo de la consulta": {{
    "Puntos totales": 5,
    "Subpuntos": {{
      "Por qué acude a la consulta": {{
        "PuntosPregunta": 5
      }}
    }}
  }},
  "Historia del proceso actual": {{
    "Puntos totales": 5,
    "Subpuntos": {{
      "Inicio del cuadro (Fecha, hora y modo brusco/progresivo)": {{
        "PuntosPregunta": 1
      }},
      "Lateralidad (Unilateral / Bilateral)": {{
        "PuntosPregunta": 1
      }},
      "Duración de los síntomas (horas, días)": {{
        "PuntosPregunta": 1
      }},
      "Evolución (Estable, empeoramiento o mejoría parcial)": {{
        "PuntosPregunta": 1
      }},
      "Episodios parecidos anteriormente": {{
        "PuntosPregunta": 1
      }}
    }}
  }},
  "Síntomas acompañantes": {{
    "Puntos totales": 16,
    "Subpuntos": {{
      "Otalgia y otorrea": {{
        "PuntosPregunta": 1
      }},
      "Dolor retroauricular": {{
        "PuntosPregunta": 1
      }},
      "Audición (Hipoacusia / Hiperacusia)": {{
        "PuntosPregunta": 2
      }},
      "Alteración del gusto (2/3 anteriores de la lengua)": {{
        "PuntosPregunta": 2
      }},
      "Disminución de la salivación": {{
        "PuntosPregunta": 2
      }},
      "Ojo seco o lagrimeo excesivo": {{
        "PuntosPregunta": 2
      }},
      "Erupción vesicular": {{
        "PuntosPregunta": 2
      }},
      "Alteración Salivación (Sialorrea / Sequedad bucal)": {{
        "PuntosPregunta": 2
      }},
      "Sensación de plenitud auricular": {{
        "PuntosPregunta": 2
      }}
    }}
  }},
  "Factores desencadenantes": {{
    "Puntos totales": 7,
    "Subpuntos": {{
      "Infección viral reciente (VHS, VZV, COVID)": {{
        "PuntosPregunta": 2
      }},
      "Exposición al frío o corrientes de aire": {{
        "PuntosPregunta": 2
      }},
      "Estrés intenso": {{
        "PuntosPregunta": 1
      }},
      "Cirugía reciente": {{
        "PuntosPregunta": 1
      }},
      "Traumatismo craneoencefálico": {{
        "PuntosPregunta": 1
      }}
    }}
  }},
  "Síntomas generales y neurológicos": {{
    "Puntos totales": 5,
    "Subpuntos": {{
      "Fiebre": {{
        "PuntosPregunta": 1
      }},
      "Malestar general": {{
        "PuntosPregunta": 1
      }},
      "Síntomas neurológicos": {{
        "PuntosPregunta": 1
      }},
      "Cefalea": {{
        "PuntosPregunta": 1
      }},
      "Vértigo": {{
        "PuntosPregunta": 1
      }}
    }}
  }},
  "Antecedentes personales y salud": {{
    "Puntos totales": 11,
    "Subpuntos": {{
      "Episodios previos de parálisis facial": {{
        "PuntosPregunta": 2
      }},
      "Síntomas respiratorios previos": {{
        "PuntosPregunta": 2
      }},
      "Enfermedades Previas (DM, HTA, Autoinmunes, VIH, Sarcoidosis, Embarazo)": {{
        "PuntosPregunta": 2
      }},
      "Alergias": {{
        "PuntosPregunta": 1
      }},
      "Intervenciones Quirúrgicas": {{
        "PuntosPregunta": 2
      }},
      "Toma de Medicamentos": {{
        "PuntosPregunta": 2
      }}
    }}
  }},
  "Medio Laboral": {{
    "Puntos totales": 6,
    "Subpuntos": {{
      "Profesión y Situación Laboral": {{
        "PuntosPregunta": 1
      }},
      "Exposición a Frío": {{
        "PuntosPregunta": 2
      }},
      "Exposición a Vibraciones": {{
        "PuntosPregunta": 1
      }},
      "Exposición a Tóxicos (Metales, Solventes, Plaguicidas)": {{
        "PuntosPregunta": 2
      }}
    }}
  }},
  "Hábitos y Antecedentes familiares": {{
    "Puntos totales": 8,
    "Subpuntos": {{
      "Hábitos (Tabaco, Alcohol, Drogas)": {{
        "PuntosPregunta": 3
      }},
      "Enfermedades Hereditarias (Diabetes, HTA)": {{
        "PuntosPregunta": 2
      }},
      "Parálisis facial recurrente en la familia": {{
        "PuntosPregunta": 1
      }},
      "Enfermedades neurológicas familiares": {{
        "PuntosPregunta": 1
      }},
      "Enfermedades hereditarias generales": {{
        "PuntosPregunta": 1
      }}
    }}
  }}
}}
Ahi te va la conversación:
    {texto_estructurado}
    """
    
    respuesta = ollama.chat(model=modelo_llm, messages=[
        {'role': 'user', 'content': prompt}
    ])
    
    return respuesta['message']['content']


# ==========================================
# FLUJO DE EJECUCIÓN PRINCIPAL
# ==========================================
if __name__ == "__main__":
    # 1. Definir directorios
    directorio_audios = "./audios"
    directorio_referencias = "./textos_referencia"
    
    modelo_diarizacion = "gpt-oss:120b-cloud"
    modelo_evaluador = "qwen3-vl:235b-cloud"
    
    # 2. Obtener lista de todos los archivos .wav en la carpeta
    archivos_audio = glob.glob(os.path.join(directorio_audios, "*.mp3"))
    
    if not archivos_audio:
        print("No se encontraron archivos de audio en el directorio especificado.")
    
    # 3. Bucle para procesar cada examen automáticamente
    for ruta_audio in archivos_audio:
        # Extraer el nombre del archivo (ej: "audio_6" de "audio_6.wav")
        nombre_base = os.path.basename(ruta_audio).replace(".wav", "")
        
        # Buscar el archivo de texto de referencia correspondiente
        ruta_referencia = os.path.join(directorio_referencias, f"{nombre_base}.txt")
        
        print(f"\n{'='*50}")
        print(f" PROCESANDO EXAMEN: {nombre_base} ")
        print(f"{'='*50}")
        
        try:
            # Leer el Ground Truth
            with open(ruta_referencia, 'r', encoding='utf-8') as f:
                transcripcion_ground_truth = f.read()
                
            # --- Ejecutar el pipeline ---
            texto_generado = transcribir_audio(ruta_audio)
            
            porcentaje_wer = calcular_wer(transcripcion_ground_truth, texto_generado)
            print(f"[!] Tasa WER para {nombre_base}: {porcentaje_wer:.2f}%")
            
            texto_diarizado = diarizar_con_ollama(texto_generado, modelo_diarizacion)
            evaluacion_final = evaluar_consulta_con_ollama(texto_diarizado, modelo_evaluador)
            
            # 4. Guardar los resultados en un archivo en lugar de solo imprimirlos
            ruta_resultado = f"./resultados/resultado_{nombre_base}.txt"
            os.makedirs(os.path.dirname(ruta_resultado), exist_ok=True)
            
            with open(ruta_resultado, 'w', encoding='utf-8') as out_file:
                out_file.write(f"--- RESULTADOS {nombre_base} ---\n")
                out_file.write(f"WER: {porcentaje_wer:.2f}%\n\n")
                out_file.write(f"--- EVALUACIÓN CLÍNICA ---\n{evaluacion_final}\n\n")
                out_file.write(f"--- DIARIZACIÓN ---\n{texto_diarizado}\n")
                
            print(f"[✓] Procesamiento de {nombre_base} completado y guardado.")

        except FileNotFoundError:
            print(f"[X] Error: No se encontró el archivo de referencia {ruta_referencia}")
        except Exception as e:
            print(f"[X] Error procesando {nombre_base}: {e}")