import os
import glob
import sys
import json
import re
import whisper
import ollama

# ==========================================
# 1. WHISPER bidezko transkripzioa
# ==========================================
def transcribir_audio(ruta_audio, modelo_whisper="medium"):
    print(f"Cargando modelo Whisper ({modelo_whisper})...")
    modelo = whisper.load_model(modelo_whisper)
    
    print("Transcribiendo audio...")
    resultado = modelo.transcribe(ruta_audio, language="es")
    return resultado["text"]

# ==========================================
# 2. Akatsen kalkulua (WER) LLM bidez
# ==========================================
def _extraer_json_desde_respuesta(texto_respuesta):
  texto_limpio = texto_respuesta.strip()

  # Saiatu zuzenean JSON gisa kargatzen
  try:
    return json.loads(texto_limpio)
  except json.JSONDecodeError:
    pass

  # Saiatu markdown kode-bloke bateko JSONa ateratzen
  bloque = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", texto_limpio, re.DOTALL)
  if bloque:
    return json.loads(bloque.group(1))

  # Azken saiakera: lehenengo { ... } blokea
  inicio = texto_limpio.find("{")
  fin = texto_limpio.rfind("}")
  if inicio != -1 and fin != -1 and fin > inicio:
    return json.loads(texto_limpio[inicio:fin + 1])

  raise ValueError("No se pudo extraer un JSON válido de la respuesta del modelo.")


def _dividir_en_bloques(texto, max_palabras=220):
  palabras = texto.split()
  if not palabras:
    return [""]

  return [" ".join(palabras[i:i + max_palabras]) for i in range(0, len(palabras), max_palabras)]


def _plantilla_resultado_wer(modelo_llm, error=None):
  resultado = {
    "modelo": modelo_llm,
    "total_palabras_referencia": 0,
    "sustituciones": 0,
    "inserciones": 0,
    "eliminaciones": 0,
    "errores_graves": 0,
    "errores_leves": 0,
    "criterio_gravedad": "sin evaluar",
    "wer_porcentaje": 0.0,
    "explicacion_breve": "",
    "ejemplos": [],
  }

  if error:
    resultado["error"] = str(error)

  return resultado


def _sumar_resultados_wer(resultados, modelo_llm):
  total_palabras = sum(int(r.get("total_palabras_referencia", 0)) for r in resultados)
  sustituciones = sum(int(r.get("sustituciones", 0)) for r in resultados)
  inserciones = sum(int(r.get("inserciones", 0)) for r in resultados)
  eliminaciones = sum(int(r.get("eliminaciones", 0)) for r in resultados)
  errores_graves = sum(int(r.get("errores_graves", 0)) for r in resultados)
  errores_leves = sum(int(r.get("errores_leves", 0)) for r in resultados)

  ejemplos = []
  for resultado in resultados:
    ejemplos.extend(resultado.get("ejemplos", []))

  wer_porcentaje = 0.0
  if total_palabras > 0:
    wer_porcentaje = ((sustituciones + inserciones + eliminaciones) / total_palabras) * 100.0

  criterio_gravedad = "errores agregados por bloques"
  if errores_graves == 0 and errores_leves == 0:
    criterio_gravedad = "sin errores detectados en los bloques"

  return {
    "modelo": modelo_llm,
    "total_palabras_referencia": total_palabras,
    "sustituciones": sustituciones,
    "inserciones": inserciones,
    "eliminaciones": eliminaciones,
    "errores_graves": errores_graves,
    "errores_leves": errores_leves,
    "criterio_gravedad": criterio_gravedad,
    "wer_porcentaje": wer_porcentaje,
    "explicacion_breve": "Resultado agregado por bloques para reducir el tamaño del prompt.",
    "ejemplos": ejemplos[:12],
  }


def _evaluar_wer_bloque(transcripcion_manual, transcripcion_automatica, modelo_llm, indice_bloque=None):
  etiqueta_bloque = f"BLOQUE {indice_bloque}" if indice_bloque is not None else "TEXTO COMPLETO"
  prompt_wer = f"""
Eres un evaluador de reconocimiento de voz en español.
Compara TRANSCRIPCION_REFERENCIA y TRANSCRIPCION_AUTOMATICA del mismo bloque.

Devuelve SOLO JSON válido con este formato exacto:
{{
  "modelo": "{modelo_llm}",
  "total_palabras_referencia": <int>,
  "sustituciones": <int>,
  "inserciones": <int>,
  "eliminaciones": <int>,
  "errores_graves": <int>,
  "errores_leves": <int>,
  "criterio_gravedad": "<string breve>",
  "wer_porcentaje": <float>,
  "explicacion_breve": "<string breve>",
  "ejemplos": [
    {{"tipo": "sustitucion|insercion|eliminacion", "gravedad": "grave|leve", "referencia": "...", "hipotesis": "..."}}
  ]
}}

Reglas:
1) WER = (sustituciones + inserciones + eliminaciones) / total_palabras_referencia * 100
2) Si no hay errores, wer_porcentaje=0.0 y ejemplos puede estar vacío.
3) No inventes contexto clínico.
4) Devuelve solo JSON válido.

{etiqueta_bloque}
TRANSCRIPCION_REFERENCIA:
{transcripcion_manual}

TRANSCRIPCION_AUTOMATICA:
{transcripcion_automatica}
"""

  respuesta = ollama.chat(model=modelo_llm, messages=[
    {'role': 'user', 'content': prompt_wer}
  ])

  return _extraer_json_desde_respuesta(respuesta['message']['content'])


def _evaluar_wer_con_modelo_por_bloques(transcripcion_manual, transcripcion_automatica, modelo_llm, max_palabras=220):
  bloques_manual = _dividir_en_bloques(transcripcion_manual, max_palabras=max_palabras)
  bloques_automatica = _dividir_en_bloques(transcripcion_automatica, max_palabras=max_palabras)
  total_bloques = max(len(bloques_manual), len(bloques_automatica))

  resultados_bloques = []
  for indice in range(total_bloques):
    bloque_manual = bloques_manual[indice] if indice < len(bloques_manual) else ""
    bloque_auto = bloques_automatica[indice] if indice < len(bloques_automatica) else ""

    resultado_bloque = _evaluar_wer_bloque(bloque_manual, bloque_auto, modelo_llm, indice + 1)
    resultados_bloques.append(resultado_bloque)

  return _sumar_resultados_wer(resultados_bloques, modelo_llm)


def _evaluar_wer_con_un_modelo(transcripcion_manual, transcripcion_automatica, modelo_llm):
  try:
    if len(transcripcion_manual.split()) > 220 or len(transcripcion_automatica.split()) > 220:
      return _evaluar_wer_con_modelo_por_bloques(transcripcion_manual, transcripcion_automatica, modelo_llm)

    return _evaluar_wer_bloque(transcripcion_manual, transcripcion_automatica, modelo_llm)
  except Exception as error:
    print(f"[!] WER con {modelo_llm} falló: {error}")
    return _plantilla_resultado_wer(modelo_llm, error=error)


def calcular_wer_con_llm(transcripcion_manual, transcripcion_automatica, modelo_llm_1, modelo_llm_2):
  print(f"Calculando WER con LLM 1 ({modelo_llm_1})...")
  resultado_1 = _evaluar_wer_con_un_modelo(transcripcion_manual, transcripcion_automatica, modelo_llm_1)

  print(f"Calculando WER con LLM 2 ({modelo_llm_2})...")
  resultado_2 = _evaluar_wer_con_un_modelo(transcripcion_manual, transcripcion_automatica, modelo_llm_2)

  wer_1 = float(resultado_1.get("wer_porcentaje", 0.0))
  wer_2 = float(resultado_2.get("wer_porcentaje", 0.0))

  return {
    "modelo_1": resultado_1,
    "modelo_2": resultado_2,
    "wer_promedio": (wer_1 + wer_2) / 2.0
  }

# ==========================================
# 3. OLLAMA erabiliz diarizazioa
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
# 4. OLLAMA bidezko auto-ebaluazioa
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
# PROGRAMAREN EXEKUZIO FLUXUA
# ==========================================
if __name__ == "__main__":
    # 1. Karpetak definitu
    directorio_audios = "./audios"
    directorio_referencias = "./textos_referencia"
    
    modelo_diarizacion = "gpt-oss:120b-cloud"
    modelo_evaluador = "qwen3-vl:235b-cloud"
    
    # 2. Karpetako .mp3 fitxategi guztien zerrenda lortu
    archivos_audio = glob.glob(os.path.join(directorio_audios, "*.mp3"))
    
    if not archivos_audio:
        print("No se encontraron archivos de audio en el directorio especificado.")
        sys.exit(0)

    # 3. Azterketa bakoitza automatikoki prozesatzeko begizta
    for ruta_audio in archivos_audio:
        # Fitxategi-izena atera (adibidez: "audio_6" fitxategitik "audio_6.mp3")
        nombre_base = os.path.basename(ruta_audio).replace(".mp3", "")

        # Erreferentzia testu-fitxategia bilatu
        ruta_referencia = os.path.join(directorio_referencias, f"{nombre_base}.txt")

        print(f"\n{'='*50}")
        print(f" PROCESANDO EXAMEN: {nombre_base} ")
        print(f"{'='*50}")

        try:
            # Erreferentzia-transkripzioa irakurri
            with open(ruta_referencia, 'r', encoding='utf-8') as f:
                transcripcion_ground_truth = f.read()

            # --- Pipeline-a exekutatu ---
            texto_generado = transcribir_audio(ruta_audio)

            resultado_wer = calcular_wer_con_llm(
              transcripcion_ground_truth,
              texto_generado,
              modelo_diarizacion,
              modelo_evaluador
            )
            porcentaje_wer = resultado_wer["wer_promedio"]
            print(f"[!] Tasa WER promedio para {nombre_base}: {porcentaje_wer:.2f}%")

            texto_diarizado = diarizar_con_ollama(texto_generado, modelo_diarizacion)
            evaluacion_final = evaluar_consulta_con_ollama(texto_diarizado, modelo_evaluador)

            # 4. Emaitzak fitxategi batean gorde, pantailan bakarrik erakutsi ordez
            ruta_resultado = f"./resultados/resultado_{nombre_base}.txt"
            os.makedirs(os.path.dirname(ruta_resultado), exist_ok=True)

            with open(ruta_resultado, 'w', encoding='utf-8') as out_file:
                out_file.write(f"--- RESULTADOS {nombre_base} ---\n")
                out_file.write(f"WER promedio (2 LLM): {porcentaje_wer:.2f}%\n\n")
                out_file.write("--- DESGLOSE WER (MODELO 1) ---\n")
                out_file.write(json.dumps(resultado_wer["modelo_1"], ensure_ascii=False, indent=2))
                out_file.write("\n\n")
                out_file.write("--- DESGLOSE WER (MODELO 2) ---\n")
                out_file.write(json.dumps(resultado_wer["modelo_2"], ensure_ascii=False, indent=2))
                out_file.write("\n\n")
                out_file.write(f"--- EVALUACIÓN CLÍNICA ---\n{evaluacion_final}\n\n")
                out_file.write(f"--- DIARIZACIÓN ---\n{texto_diarizado}\n")

            print(f"[✓] Procesamiento de {nombre_base} completado y guardado.")

        except FileNotFoundError:
            print(f"[X] Error: No se encontró el archivo de referencia {ruta_referencia}")
        except Exception as e:
            print(f"[X] Error procesando {nombre_base}: {e}")