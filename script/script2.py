import os
import sys
from pathlib import Path
import ollama
import whisper

def transcribe_audio(audio_path, whisper_model):
   print(f"Whisper eredua kargatzen ({whisper_model})...")
   model = whisper.load_model(whisper_model)

   print("Audioa transkribatzen...")
   result = model.transcribe(str(audio_path), language="es", fp16=False)
   return result['text']

def diarize_text(plain_text, llm_model):
   print(f"Diarizazioa abiarazten Ollama-rekin (eredua: {llm_model})...")

   prompt_diarization = f"""
   Te voy a mandar una conversación entre medico y paciente, y necesito que me digas quien de los dos es el que
   esta diciendo cada linea de la conversación. Como output, quiero que me devuelvas la propia conversación pero
   acotando al principio de cada linea si es el MEDICO: o el PACIENTE: el que la dice.
   Ahi te va la conversación:

    {plain_text}
    """
    
   response = ollama.chat(model=llm_model, messages=[
        {'role': 'user', 'content': prompt_diarization}
    ])
    
   return response['message']['content']

def calculate_wer_with_llm(reference_text, transcription, llm_model):
   print("WER kalkulatzen LLM batekin...")

   prompt = f"""
   Necesito que calcules el Word Error Rate de la siguiente transcripcion:
   ###
   {transcription}
   ###
   Teniendo en cuenta que el texto referencia es el siguiente:

   ###
   {reference_text}
   ###

   Como respuesta, devuelveme el porcentaje de WER, y debajo, el desglose de los errores que se han cometido (palabras correctas, palabras mal transcritas, palabras añadidas y palabras omitidas).
   """

   response = ollama.chat(
      model = llm_model, messages=[
         {'role':'user', 'content': prompt}
      ]
   )

   return response['message']['content']

def evaluate(texto_estructurado, llm_model):
   print(f"Ebaluazio klinikoa hasten Ollama-rekin (eredua: {llm_model})...")

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

   response = ollama.chat(model=llm_model, messages=[
      {'role': 'user', 'content': prompt}
   ])
   
   return response['message']['content']

if __name__ == "__main__":
   if len(sys.argv) < 3:
        print("Erabilera: python script.py <audio_karpeta> <erreferentzia_karpeta>")
        sys.exit(1)

   audio_directory = sys.argv[1]
   reference_directory = sys.argv[2]

   diarization_model = 'gemma3:12b-cloud'
   evaluation_model = 'gpt-oss:120b-cloud'

   audios = Path(audio_directory)

   for audio_file in audios.iterdir():
      base_name = audio_file.stem

      reference_path = os.path.join(reference_directory, f"{base_name}.txt")

      print(f"\n{'='*50}")
      print(f" PROZESATZEN: {base_name} ")
      print(f"{'='*50}")

      try:
         with open(reference_path, 'r', encoding='utf-8') as f:
            reference_transcription = f.read()

         generated_text = transcribe_audio(audio_file, 'medium')

         wer_percentage = calculate_wer_with_llm(reference_transcription, generated_text, 'gpt-oss:120b-cloud') #qwen3-vl:235b-cloud
         print(f"WER ehunekoa {base_name} transkribaketarako: {wer_percentage}")
         
         diarized_text = diarize_text(generated_text, diarization_model)
         final_evaluation = evaluate(diarized_text, evaluation_model)

         result_path = f"./resultados_castellano2/resultado_{base_name}.txt"
         os.makedirs(os.path.dirname(result_path), exist_ok=True)
         
         with open(result_path, 'w', encoding='utf-8') as out_file:
            out_file.write(f"--- EMAITZAK {base_name} ---\n")
            out_file.write(f"WER: {wer_percentage}\n\n")
            out_file.write(f"--- EBALUAZIO KLINIKOA ---\n{final_evaluation}\n\n")
            out_file.write(f"--- DIARIZAZIOA ---\n{diarized_text}\n")

         print(f"{base_name} elkarrizketaren prozesamendua amaitu eta gorde da.")
         
      except FileNotFoundError:
         print(f"Errorea: ez da aurkitu erreferentzia-fitxategia {reference_path}")
      except Exception as e:
         print(f"Errorea {base_name} prozesatzean: {e}")