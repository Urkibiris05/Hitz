import argparse
from pathlib import Path

import whisper


def transcribir_audio(ruta_audio, modelo_whisper, output_dir):
    resultado = modelo_whisper.transcribe(
        str(ruta_audio),
        language="es",
        verbose=True,
        output_dir=str(output_dir),
        output_format="txt"
    )
    return resultado["text"].strip()


def procesar_carpeta(audio_dir, output_dir, nombre_modelo):
    carpeta_audio = Path(audio_dir)
    carpeta_salida = Path(output_dir)
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    modelo = whisper.load_model(nombre_modelo)
    print(f"Modelo Whisper cargado: {nombre_modelo}")

    extensiones_validas = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".mp4", ".mpeg", ".mpga"}
    audios = [
        archivo
        for archivo in sorted(carpeta_audio.iterdir())
        if archivo.is_file() and archivo.suffix.lower() in extensiones_validas
    ]

    if not audios:
        print(f"No se han encontrado audios válidos en {carpeta_audio}")
        return

    total = len(audios)

    for indice, audio_file in enumerate(audios, start=1):
        print(f"[{indice}/{total}] Transcribiendo {audio_file.name}...")
        texto = transcribir_audio(audio_file, modelo, output_dir)

        salida = carpeta_salida / f"{audio_file.stem}.txt"
        salida.write_text(texto + "\n", encoding="utf-8")
        print(f"Guardado en {salida} (solo texto plano)")


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe todos los audios de una carpeta con Whisper y guarda un .txt por audio."
    )
    parser.add_argument("audio_dir", help="Carpeta con los audios a transcribir")
    parser.add_argument("output_dir", help="Carpeta donde guardar las transcripciones")
    parser.add_argument(
        "--model",
        default="medium",
        help="Modelo Whisper a usar (por defecto: medium)",
    )

    args = parser.parse_args()
    procesar_carpeta(args.audio_dir, args.output_dir, args.model)


if __name__ == "__main__":
    main()