import os
import sys
from pathlib import Path
import whisper
import ollama

def transcribir_audio(ruta_audio, modelo_whisper="medium"):
   print(f"Cargando modelo Whisper ({modelo_whisper})...")
   modelo = whisper.load_model(modelo_whisper)

   print("Transcribiendo audio...")
   resultado = modelo.transcribe(str(ruta_audio), language="es", fp16=False)
   return resultado['text']

def diarizar_texto(texto_plano, modelo_llm="gpt-oss:20b-cloud"):
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

def calcular_wer_llm(texto_referencia, transcripcion, modelo_llm):
   print("Calculando Tasa de Error de Palabras (WER) con un LLM...")

   prompt = f"""
   Necesito que calcules el Word Error Rate de la siguiente transcripcion:
   ###
   {transcripcion}
   ###
   Teniendo en cuenta que el texto referencia es el siguiente:

   ###
   {texto_referencia}
   ###

   Como respuesta devuelveme solo el porcentaje, no quiero ningun calculo:
   """

   respuesta = ollama.chat(
      model = modelo_llm, messages=[
         {'role':'user', 'content': prompt}
      ]
   )

   return respuesta['message']['content']

def evaluar_conversacion(texto_estructurado, modelo_llm='gemma3:12b-cloud'):
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

if __name__ == "__main__":
   if len(sys.argv) < 3:
        print("Uso: python script.py <directorio_audios> <directorio_referencias>")
        sys.exit(1)

   directorio_audios = sys.argv[1]
   directorio_referencias = sys.argv[2]

   modelo_diarizacion = 'gemma3:12b-cloud'
   modelo_evaluador = 'gpt-oss:120b-cloud'

   audios = Path(directorio_audios)

   for audio in audios.iterdir():
      nombre_base = audio.stem

      ruta_referencia = os.path.join(directorio_referencias, f"{nombre_base}.txt")

      print(f"\n{'='*50}")
      print(f" PROCESANDO EXAMEN: {nombre_base} ")
      print(f"{'='*50}")

      try:
         with open(ruta_referencia, 'r', encoding='utf-8') as f:
            transcripcion_gorund_truth = f.read()

         texto_generado = transcribir_audio(audio)

         porcentaje_wer = calcular_wer_llm(transcripcion_gorund_truth, texto_generado, 'qwen3-vl:235b-cloud')
         print(f"Tasa WER para {nombre_base}: {porcentaje_wer}")
         
         texto_diarizado = diarizar_texto(texto_generado, modelo_diarizacion)
         evaluacion_final = evaluar_conversacion(texto_diarizado, modelo_evaluador)

         ruta_resultado = f"./resultados2/resultado_{nombre_base}.txt"
         os.makedirs(os.path.dirname(ruta_resultado), exist_ok=True)
         
         with open(ruta_resultado, 'w', encoding='utf-8') as out_file:
            out_file.write(f"--- RESULTADOS {nombre_base} ---\n")
            out_file.write(f"WER: {porcentaje_wer}\n\n")
            out_file.write(f"--- EVALUACIÓN CLÍNICA ---\n{evaluacion_final}\n\n")
            out_file.write(f"--- DIARIZACIÓN ---\n{texto_diarizado}\n")

         print(f"Procesamiento de {nombre_base} completado y guardado.")
         
      except FileNotFoundError:
         print(f"Error: No se encontró el archivo de referencia {ruta_referencia}")
      except Exception as e:
         print(f"Error procesando {nombre_base}: {e}")