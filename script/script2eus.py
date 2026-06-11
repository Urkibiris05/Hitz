import os
import sys
from pathlib import Path
import whisper
import ollama

def transcribe_audio(audio_path, whisper_model="medium"):
   print(f"Whisper eredua kargatzen ({whisper_model})...")
   model = whisper.load_model(whisper_model)

   print("Audioa transkribitzen...")
   result = model.transcribe(str(audio_path), language="eu", fp16=False)
   return result['text']

def diarize_text(plain_text, llm_model):
   print(f"Diarizazioa hasten Ollamarekin (Eredua: {llm_model})...")

   prompt = f"""
   Medikuaren eta pazientearen arteko elkarrizketa bat bidaliko dizut, eta esan behar didazu bietako nor ari den hizketan elkarrizketa-esaldi bakoitzean. Output gisa, elkarrizketa bera itzultzea nahi dut, baina lerro bakoitzaren hasieran mugatuta, MEDIKUA bada: edo PAZIENTEA bada: esaten duena. 
   Hor doakizu elkarrizketa:



   {plain_text}
    """
    
   result = ollama.chat(model=llm_model, messages=[
        {'role': 'user', 'content': prompt}
    ])
    
   return result['message']['content']

def calculate_wer_llm(reference_text, transcription, llm_model):
   print("Calculando Tasa de Error de Palabras (WER) con un LLM...")

   prompt = f"""
   Transkripzio honen Word Error Rate kalkulatzea behar dut:
   ###
   {transcription}
   ###
   Erreferentziako testua honako hau dela kontuan hartuta:

   ###
   {reference_text}
   ###

   Erantzun bezala, itzul iezadazu ehunekoa bakarrik, ez dut kalkulurik nahi:
   """

   result = ollama.chat(
      model = llm_model, messages=[
         {'role':'user', 'content': prompt}
      ]
   )

   return result['message']['content']

def evaluate_conversation(structured_text, llm_model):
   print(f"Ebaluazio klinikoa hasten Ollamarekin (Eredua: {llm_model})...")

   prompt = f"""Paziente bati kontsultatzeko garaian, mediku batek bere protokoloa behar bezala betetzen duen ebaluatzeko balio duten irizpide batzuk bidaliko dizkizut. Irizpideekin batera, mediku baten eta paziente baten arteko elkarrizketa bidaliko dizut. Irizpideetan oinarrituta, medikuak elkarrizketa honetan lortzen duen puntuazioa ebaluatzea nahi dut.


   Irizpideak:
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
   {structured_text}
   """

   result = ollama.chat(model=llm_model, messages=[
        {'role': 'user', 'content': prompt}
    ])
   
   return result['message']['content']

if __name__ == "__main__":
   if len(sys.argv) < 3:
        print("Erabilera: python script2eus.py <audioen_direktorioa> <erreferentzien_direktorioa>")
        sys.exit(1)

   audio_directory = sys.argv[1]
   reference_directory = sys.argv[2]

   diarization_model = 'gemma3:12b-cloud'
   evaluation_model = 'qwen3-vl:235b-cloud'

   audio_files = Path(audio_directory)

   for audio in audio_files.iterdir():
      base_name = audio.stem

      reference_path = os.path.join(reference_directory, f"{base_name}.txt")

      print(f"\n{'='*50}")
      print(f" AZTERKETA PROZESATZEN: {base_name} ")
      print(f"{'='*50}")

      try:
         with open(reference_path, 'r', encoding='utf-8') as f:
            ground_truth_transcription = f.read()

         generated_text = transcribe_audio(audio)

         wer_percentage = calculate_wer_llm(ground_truth_transcription, generated_text, 'gpt-oss:120b-cloud')
         print(f"WER TASA {base_name}-rentzat: {wer_percentage}")
         
         diarized_text = diarize_text(generated_text, diarization_model)
         final_evaluation = evaluate_conversation(diarized_text, evaluation_model)

         results_path = f"./emaitzak_euskera/resultado_{base_name}.txt"
         os.makedirs(os.path.dirname(results_path), exist_ok=True)
         
         with open(results_path, 'w', encoding='utf-8') as out_file:
            out_file.write(f"--- EMAITZAK {base_name} ---\n")
            out_file.write(f"WER: {wer_percentage}\n\n")
            out_file.write(f"--- EBALUAKETA KLINIKOA ---\n{final_evaluation}\n\n")
            out_file.write(f"--- DIARIZAZIOA ---\n{diarized_text}\n")

         print(f"{base_name}-ren prozesamendua amaituta eta gordeta.")
         
      except FileNotFoundError:
         print(f"Errorea: Ez da aurkitu erreferentzia-fitxategia {reference_path}")
      except Exception as e:
         print(f"Errorea {base_name} prozesatzen: {e}")