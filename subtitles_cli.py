#!/usr/bin/env python3
"""
subtitles_cli.py
================
CLI para generacion de subtitulos .srt desde audio/video.

Uso:
    python subtitles_cli.py video.mp4
    python subtitles_cli.py audio.wav --language en
    python subtitles_cli.py video.mp4 --output subtitulos.srt --translate es
    python subtitles_cli.py video.mp4 --model large-v2 --diarize
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="subtitles_cli",
        description="Genera subtitulos .srt desde audio/video con IA local",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python subtitles_cli.py video.mp4
  python subtitles_cli.py audio.wav --language en
  python subtitles_cli.py video.mp4 --output subs.srt --model large-v2
  python subtitles_cli.py video.mp4 --translate es
  python subtitles_cli.py video.mp4 --diarize --no-denoise
  python subtitles_cli.py video.mp4 --gguf ~/.subtitle_ai/models/llama3.gguf
        """,
    )

    parser.add_argument("input", help="Archivo de audio o video (mp4, mkv, mp3, wav, m4a...)")
    parser.add_argument(
        "-o", "--output",
        help="Archivo .srt de salida (default: mismo nombre que la entrada)",
    )

    # ASR
    asr = parser.add_argument_group("Transcripcion (Whisper)")
    asr.add_argument(
        "-l", "--language", default="auto",
        help="Idioma del audio: es, en, fr, de, auto... (default: auto)",
    )
    asr.add_argument(
        "-m", "--model", default="medium",
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        help="Modelo Whisper (default: medium)",
    )

    # Audio
    audio = parser.add_argument_group("Procesamiento de audio")
    audio.add_argument("--no-denoise", action="store_true", help="Desactivar reduccion de ruido")
    audio.add_argument("--no-quality-check", action="store_true", help="Omitir diagnostico de calidad")

    # LLM
    llm = parser.add_argument_group("Motor LLM (correccion y traduccion)")
    llm.add_argument("--no-correct", action="store_true", help="Desactivar correccion LLM")
    llm.add_argument(
        "--translate", metavar="LANG",
        help="Traducir subtitulos al idioma especificado (ej: es, en, fr)",
    )
    llm.add_argument(
        "--no-auto-translate", action="store_true",
        help="No traducir automaticamente al espanol si el audio esta en otro idioma",
    )
    llm.add_argument(
        "--gguf", metavar="PATH",
        help="Ruta a modelo GGUF para llama.cpp (auto-detectado si no se indica)",
    )
    llm.add_argument(
        "--llm-model", default="llama3", metavar="MODEL",
        help="Modelo Ollama como fallback (default: llama3)",
    )

    # Diarizacion
    dia = parser.add_argument_group("Diarizacion de hablantes")
    dia.add_argument("--diarize", action="store_true", help="Identificar hablantes (requiere HF_TOKEN)")
    dia.add_argument("--hf-token", metavar="TOKEN", help="Token de HuggingFace para pyannote")

    # Misc
    parser.add_argument("-v", "--verbose", action="store_true", help="Mostrar logs detallados")

    return parser


def print_banner():
    print("""
  ╔══════════════════════════════════════════════════╗
  ║          SubtitleAI — Generador de SRT           ║
  ║    faster-whisper + llama.cpp + pyannote         ║
  ╚══════════════════════════════════════════════════╝
""")


def make_progress_bar(step: str, pct: int):
    filled = pct // 5
    bar = "\u2588" * filled + "\u2591" * (20 - filled)
    print(f"\r  [{bar}] {pct:3d}%  {step:<48}", end="", flush=True)


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"\n  Error: archivo no encontrado: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output) if args.output else input_path.with_suffix(".srt")

    print_banner()
    print(f"  Entrada : {input_path}")
    print(f"  Salida  : {output_path}")
    print(f"  Idioma  : {args.language}  |  Modelo: {args.model}")
    if args.translate:
        print(f"  Traducir: {args.translate}")
    print()

    try:
        from core.pipeline import run_pipeline, segments_to_srt
    except ImportError as e:
        print(f"\n  Error importando el pipeline: {e}", file=sys.stderr)
        print("  Ejecuta primero: python setup_blackwell.py", file=sys.stderr)
        sys.exit(1)

    result = run_pipeline(
        audio_path=str(input_path),
        language=args.language,
        model_size=args.model,
        llm_model=args.llm_model,
        gguf_model_path=args.gguf,
        enable_denoise=not args.no_denoise,
        enable_diarization=args.diarize,
        enable_correction=not args.no_correct,
        enable_translation=bool(args.translate),
        target_language=args.translate,
        auto_translate_to_spanish=not args.no_auto_translate,
        hf_token=args.hf_token,
        progress_callback=make_progress_bar,
    )

    print()  # salto de linea tras la barra de progreso

    srt_content = segments_to_srt(result.segments)
    output_path.write_text(srt_content, encoding="utf-8")

    print(f"\n  Segmentos  : {len(result.segments)}")
    print(f"  Idioma     : {result.language} ({result.language_probability:.0%})")

    if result.audio_quality:
        q = result.audio_quality
        print(f"  Calidad    : {q.quality_label}  (SNR {q.snr_db} dB | voz {q.speech_ratio:.0%})")
        print(f"  Ruido      : {q.noise_profile}")

    if result.speakers_found:
        print(f"  Hablantes  : {result.speakers_found}")
    if result.corrected:
        print(f"  LLM        : correccion aplicada")
    if result.translated:
        print(f"  Traduccion : {result.target_language}")

    if result.errors:
        print("\n  Avisos:")
        for e in result.errors:
            print(f"    - {e}")

    print(f"\n  Archivo generado: {output_path}\n")


if __name__ == "__main__":
    main()
