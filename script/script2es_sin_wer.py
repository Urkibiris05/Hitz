import os
import sys
from pathlib import Path

from script2es import diarize_text, evaluate, transcribe_audio


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
            _reference_transcription = f.read()

         generated_text = transcribe_audio(audio_file, 'medium')

         diarized_text = diarize_text(generated_text, diarization_model)
         final_evaluation = evaluate(diarized_text, evaluation_model)

         result_path = f"./resultados_castellano2/resultado_{base_name}.txt"
         os.makedirs(os.path.dirname(result_path), exist_ok=True)

         with open(result_path, 'w', encoding='utf-8') as out_file:
            out_file.write(f"--- EMAITZAK {base_name} ---\n")
            out_file.write(f"--- EBALUAZIO KLINIKOA ---\n{final_evaluation}\n\n")
            out_file.write(f"--- DIARIZAZIOA ---\n{diarized_text}\n")

         print(f"{base_name} elkarrizketaren prozesamendua amaitu eta gorde da.")

      except FileNotFoundError:
         print(f"Errorea: ez da aurkitu erreferentzia-fitxategia {reference_path}")
      except Exception as e:
         print(f"Errorea {base_name} prozesatzean: {e}")