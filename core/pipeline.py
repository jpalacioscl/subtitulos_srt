"""
core/pipeline.py
================
Pipeline de IA completo para generación de subtítulos .srt
Capas: Preprocesamiento → Transcripción → Diarización → Corrección LLM → Traducción

LLM Backend: llama.cpp (principal) con Ollama como fallback automático.
"""

import os
import re
import json
import tempfile
import logging
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Estructuras de datos
# ─────────────────────────────────────────────

@dataclass
class Segment:
    index: int
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    confidence: float = 1.0

@dataclass
class PipelineResult:
    segments: list[Segment]
    language: str
    language_probability: float
    duration: float
    speakers_found: int = 0
    corrected: bool = False
    translated: bool = False
    target_language: Optional[str] = None
    errors: list[str] = field(default_factory=list)
    audio_quality: Optional[object] = None   # AudioQualityReport


# ─────────────────────────────────────────────
# Utilidades SRT
# ─────────────────────────────────────────────

def seconds_to_srt(seconds: float) -> str:
    """Convierte segundos a formato SRT: HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def segments_to_srt(segments: list[Segment]) -> str:
    """Convierte lista de segmentos a string formato .srt"""
    blocks = []
    for seg in segments:
        speaker_prefix = f"[{seg.speaker}] " if seg.speaker else ""
        block = f"{seg.index}\n{seconds_to_srt(seg.start)} --> {seconds_to_srt(seg.end)}\n{speaker_prefix}{seg.text}\n"
        blocks.append(block)
    return "\n".join(blocks)


# ─────────────────────────────────────────────
# Capa 1: Preprocesamiento de audio
# ─────────────────────────────────────────────

def preprocess_audio(input_path: str, output_path: str) -> str:
    """
    Convierte cualquier formato a WAV mono 16kHz (óptimo para Whisper)
    usando ffmpeg. Si ffmpeg no está disponible, retorna el archivo original.
    """
    try:
        import subprocess
        cmd = [
            "ffmpeg", "-i", input_path,
            "-ac", "1",           # mono
            "-ar", "16000",       # 16kHz
            "-acodec", "pcm_s16le",
            "-y",                 # sobrescribir
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            logger.info(f"[Preprocesamiento] Audio convertido: {output_path}")
            return output_path
        else:
            logger.warning(f"[Preprocesamiento] ffmpeg falló: {result.stderr}")
            return input_path
    except (FileNotFoundError, Exception) as e:
        logger.warning(f"[Preprocesamiento] No disponible: {e}")
        return input_path


def denoise_audio(audio_path: str) -> str:
    """
    Reducción de ruido con noisereduce.
    Retorna ruta al audio limpio, o el original si no está instalado.
    """
    try:
        import noisereduce as nr
        import librosa
        import soundfile as sf

        logger.info("[Preprocesamiento] Aplicando reducción de ruido...")
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        audio_clean = nr.reduce_noise(y=audio, sr=sr, stationary=False, prop_decrease=0.75)

        clean_path = audio_path.replace(".wav", "_clean.wav")
        sf.write(clean_path, audio_clean, sr)
        logger.info(f"[Preprocesamiento] Ruido reducido: {clean_path}")
        return clean_path
    except ImportError:
        logger.warning("[Preprocesamiento] noisereduce no instalado, saltando limpieza de audio.")
        return audio_path
    except Exception as e:
        logger.warning(f"[Preprocesamiento] Error en denoising: {e}")
        return audio_path


# ─────────────────────────────────────────────
# Capa 2: Transcripción ASR con faster-whisper
# ─────────────────────────────────────────────

def transcribe(audio_path: str, language: str = "es", model_size: str = "medium") -> tuple[list[Segment], dict]:
    """
    Transcribe audio con faster-whisper.
    Retorna (segmentos, info_metadata).
    """
    from faster_whisper import WhisperModel

    logger.info(f"[ASR] Cargando modelo Whisper '{model_size}'...")
    device = "cpu"
    compute = "int8"

    # Detectar GPU si está disponible
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            compute = "float16"
            logger.info("[ASR] GPU detectada, usando CUDA.")
    except ImportError:
        pass

    model = WhisperModel(model_size, device=device, compute_type=compute)

    logger.info(f"[ASR] Transcribiendo en idioma '{language}'...")
    raw_segments, info = model.transcribe(
        audio_path,
        language=language if language != "auto" else None,
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        word_timestamps=True,
    )

    segments = []
    for i, seg in enumerate(raw_segments, 1):
        segments.append(Segment(
            index=i,
            start=seg.start,
            end=seg.end,
            text=seg.text.strip(),
            confidence=seg.avg_logprob if hasattr(seg, 'avg_logprob') else 1.0,
        ))

    metadata = {
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "duration": info.duration if hasattr(info, 'duration') else 0,
    }

    logger.info(f"[ASR] Transcripción completa. {len(segments)} segmentos. Idioma: {info.language} ({info.language_probability:.0%})")
    return segments, metadata


# ─────────────────────────────────────────────
# Capa 3: Diarización de hablantes (pyannote)
# ─────────────────────────────────────────────

def diarize(audio_path: str, hf_token: Optional[str] = None) -> Optional[list[dict]]:
    """
    Identifica quién habla en cada momento.
    Requiere pyannote.audio y un token de HuggingFace.
    Si no está disponible, retorna None sin romper el pipeline.
    """
    try:
        from pyannote.audio import Pipeline
        import torch

        token = hf_token or os.environ.get("HF_TOKEN")
        if not token:
            logger.warning("[Diarización] No se encontró HF_TOKEN. Saltando diarización.")
            return None

        logger.info("[Diarización] Cargando pipeline de pyannote...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token
        )

        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        pipeline = pipeline.to(__import__("torch").device(device))

        logger.info("[Diarización] Procesando hablantes...")
        diarization = pipeline(audio_path)

        turns = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            turns.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker,
            })

        logger.info(f"[Diarización] Encontrados {len(set(t['speaker'] for t in turns))} hablantes.")
        return turns

    except ImportError:
        logger.warning("[Diarización] pyannote.audio no instalado. Instala con: pip install pyannote.audio")
        return None
    except Exception as e:
        logger.warning(f"[Diarización] Error: {e}")
        return None


def assign_speakers(segments: list[Segment], diarization: Optional[list[dict]]) -> list[Segment]:
    """
    Asigna etiquetas de hablante a cada segmento por solapamiento temporal.
    """
    if not diarization:
        return segments

    # Crear mapa de nombres más legibles: SPEAKER_00 → Hablante A
    speaker_labels = {}
    letter = ord('A')

    for seg in segments:
        best_speaker = None
        best_overlap = 0.0

        for turn in diarization:
            overlap = max(0, min(seg.end, turn["end"]) - max(seg.start, turn["start"]))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn["speaker"]

        if best_speaker:
            if best_speaker not in speaker_labels:
                speaker_labels[best_speaker] = f"Hablante {chr(letter)}"
                letter += 1
            seg.speaker = speaker_labels[best_speaker]

    logger.info(f"[Diarización] Hablantes asignados a {len(segments)} segmentos.")
    return segments


# ─────────────────────────────────────────────
# Capa 4: Corrección con LLM (llama.cpp / Ollama)
# ─────────────────────────────────────────────

def correct_with_llm(segments: list[Segment], llm_engine) -> list[Segment]:
    """
    Corrige errores de transcripción usando el LLMEngine activo.
    Soporta llama.cpp (principal) y Ollama (fallback).
    """
    if llm_engine is None or not llm_engine.is_active:
        logger.warning("[LLM] Motor LLM no disponible. Saltando corrección.")
        return segments

    corrected_segments, _ = llm_engine.correct_subtitles(segments)
    return corrected_segments


# ─────────────────────────────────────────────
# Capa 5: Traducción con LLM (llama.cpp / Ollama)
# ─────────────────────────────────────────────

def translate_with_llm(segments: list[Segment], target_language: str, llm_engine) -> list[Segment]:
    """
    Traduce subtítulos usando el LLMEngine activo.
    Soporta llama.cpp (principal) y Ollama (fallback).
    """
    if llm_engine is None or not llm_engine.is_active:
        logger.warning("[LLM] Motor LLM no disponible. Saltando traducción.")
        return segments

    return llm_engine.translate_subtitles(segments, target_language)

LANGUAGE_NAMES = {
    "es": "español", "en": "inglés", "fr": "francés",
    "de": "alemán", "it": "italiano", "pt": "portugués",
    "ja": "japonés", "zh": "chino", "ru": "ruso",
    "ar": "árabe", "ko": "coreano",
}

# Idiomas que NO son español → traducir automáticamente a español
NON_SPANISH_LANGUAGES = {
    "en", "fr", "de", "it", "pt", "ja", "zh", "ru", "ar", "ko",
    "nl", "pl", "sv", "da", "fi", "no", "tr", "cs", "ro", "hu",
    "uk", "ca", "he", "id", "ms", "th", "vi", "hi",
}


# ─────────────────────────────────────────────
# Diagnóstico de calidad de audio
# ─────────────────────────────────────────────

@dataclass
class AudioQualityReport:
    snr_db: float                  # Signal-to-Noise Ratio estimado en dB
    speech_ratio: float            # % del audio que contiene voz activa (0-1)
    duration_seconds: float        # Duración total del audio
    rms_mean: float                # Energía media RMS
    noise_floor: float             # Nivel estimado del ruido de fondo
    quality_label: str             # EXCELENTE / BUENA / ACEPTABLE / DIFÍCIL
    recommended_model: str         # Modelo Whisper recomendado
    recommended_denoise: bool      # Si se recomienda aplicar denoise
    recommended_beam_size: int     # beam_size recomendado para Whisper
    warnings: list[str]            # Advertencias específicas detectadas
    noise_profile: str             # Tipo de ruido dominante estimado


def analyze_audio_quality(audio_path: str) -> AudioQualityReport:
    """
    Analiza el audio y genera un diagnóstico completo de calidad.
    Detecta nivel de ruido, ratio de voz, tipo de ruido dominante
    y recomienda la mejor configuración del pipeline.

    Útil especialmente para audios de documentales con ruido ambiental
    (animales, viento, agua, música de fondo).
    """
    try:
        import librosa
        import numpy as np

        logger.info("[Diagnóstico] Analizando calidad del audio...")

        # Cargar audio a 16kHz mono (igual que Whisper)
        audio, sr = librosa.load(audio_path, sr=16000, mono=True, duration=120)  # max 2min de muestra
        duration = librosa.get_duration(y=audio, sr=sr)

        # ── Energía por frames ────────────────────────
        frame_length = 2048
        hop_length = 512
        frame_rms = librosa.feature.rms(
            y=audio, frame_length=frame_length, hop_length=hop_length
        )[0]

        rms_mean = float(np.mean(frame_rms))
        noise_floor = float(np.percentile(frame_rms, 10))   # percentil bajo = ruido de fondo
        voice_peak  = float(np.percentile(frame_rms, 85))   # percentil alto = voz activa

        # ── SNR estimado ──────────────────────────────
        snr_db = float(20 * np.log10(voice_peak / (noise_floor + 1e-10)))
        snr_db = max(0.0, min(snr_db, 60.0))  # clamp a rango razonable

        # ── Ratio de voz activa (VAD simple) ──────────
        vad_threshold = noise_floor * 3.5
        speech_frames = frame_rms > vad_threshold
        speech_ratio = float(speech_frames.sum() / len(speech_frames))

        # ── Análisis espectral para tipo de ruido ─────
        stft = np.abs(librosa.stft(audio, n_fft=2048))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

        # Bandas de frecuencia
        low_band    = stft[(freqs >= 20)   & (freqs < 300), :].mean()    # graves (rugidos, motores)
        mid_band    = stft[(freqs >= 300)  & (freqs < 3000), :].mean()   # medios (voz humana)
        high_band   = stft[(freqs >= 3000) & (freqs < 8000), :].mean()   # agudos (pájaros, insectos)

        # Estimar tipo de ruido dominante
        if low_band > mid_band * 1.5:
            noise_profile = "Ruido de baja frecuencia (motores, viento fuerte, rugidos)"
        elif high_band > mid_band * 1.2:
            noise_profile = "Ruido de alta frecuencia (pájaros, insectos, lluvia fina)"
        elif low_band > 0 and high_band > 0 and abs(low_band - high_band) < mid_band * 0.3:
            noise_profile = "Ruido de banda ancha (lluvia torrencial, cascada, océano)"
        else:
            noise_profile = "Ruido mixto / Ambiente natural equilibrado"

        # ── Determinar calidad y recomendaciones ──────
        warnings = []

        if snr_db >= 20:
            quality_label = "EXCELENTE"
            recommended_model = "small"
            recommended_denoise = False
            recommended_beam_size = 5
        elif snr_db >= 12:
            quality_label = "BUENA"
            recommended_model = "medium"
            recommended_denoise = True
            recommended_beam_size = 5
        elif snr_db >= 6:
            quality_label = "ACEPTABLE"
            recommended_model = "medium"
            recommended_denoise = True
            recommended_beam_size = 8
            warnings.append("Ruido significativo — el denoise mejorará la transcripción.")
        else:
            quality_label = "DIFÍCIL"
            recommended_model = "large-v2"
            recommended_denoise = True
            recommended_beam_size = 10
            warnings.append("Ruido muy alto — considera limpiar el audio manualmente con Audacity.")

        if speech_ratio < 0.25:
            warnings.append(f"Solo el {speech_ratio:.0%} del audio contiene voz — muchos silencios o ruido sin habla.")
        if speech_ratio > 0.95:
            warnings.append("Voz continua sin pausas — el VAD podría no segmentar bien.")
        if duration > 3600:
            warnings.append(f"Audio largo ({duration/3600:.1f}h) — el procesamiento tomará tiempo considerable.")

        # Detectar posible voz/canto de fondo (segunda fuente vocal)
        spectral_flatness = float(np.mean(librosa.feature.spectral_flatness(y=audio)))
        if spectral_flatness > 0.15 and snr_db < 15:
            warnings.append("Posible música con letra o segunda fuente de voz detectada — puede afectar la transcripción.")

        report = AudioQualityReport(
            snr_db=round(snr_db, 1),
            speech_ratio=round(speech_ratio, 3),
            duration_seconds=round(duration, 1),
            rms_mean=round(rms_mean, 5),
            noise_floor=round(noise_floor, 5),
            quality_label=quality_label,
            recommended_model=recommended_model,
            recommended_denoise=recommended_denoise,
            recommended_beam_size=recommended_beam_size,
            warnings=warnings,
            noise_profile=noise_profile,
        )

        logger.info(f"[Diagnóstico] SNR: {snr_db:.1f}dB | Calidad: {quality_label} | Voz: {speech_ratio:.0%}")
        return report

    except ImportError:
        logger.warning("[Diagnóstico] librosa no instalado. Instala con: pip install librosa")
        return AudioQualityReport(
            snr_db=0, speech_ratio=0, duration_seconds=0, rms_mean=0,
            noise_floor=0, quality_label="DESCONOCIDA", recommended_model="medium",
            recommended_denoise=True, recommended_beam_size=5,
            warnings=["librosa no disponible — no se pudo analizar el audio."],
            noise_profile="No analizado",
        )
    except Exception as e:
        logger.warning(f"[Diagnóstico] Error en análisis: {e}")
        return AudioQualityReport(
            snr_db=0, speech_ratio=0, duration_seconds=0, rms_mean=0,
            noise_floor=0, quality_label="DESCONOCIDA", recommended_model="medium",
            recommended_denoise=True, recommended_beam_size=5,
            warnings=[f"Error en análisis: {str(e)}"],
            noise_profile="No analizado",
        )

def translate_with_llm(segments: list[Segment], target_language: str, model: str = "llama3") -> list[Segment]:
    """
    Traduce subtítulos manteniendo timestamps exactos usando Ollama.
    """
    try:
        import requests

        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=3)
            r.raise_for_status()
        except Exception:
            logger.warning("[Traducción] Ollama no está corriendo. Saltando traducción.")
            return segments

        lang_name = LANGUAGE_NAMES.get(target_language, target_language)
        logger.info(f"[Traducción] Traduciendo a {lang_name} con modelo '{model}'...")

        BATCH_SIZE = 15
        translated_segments = []

        for i in range(0, len(segments), BATCH_SIZE):
            batch = segments[i:i + BATCH_SIZE]
            batch_text = "\n".join(
                [f"[{seg.index}] {seg.text}" for seg in batch]
            )

            prompt = f"""Eres un traductor profesional especializado en subtítulos.

Traduce al {lang_name} los siguientes subtítulos.

REGLAS ESTRICTAS:
1. Mantén el formato: [NUMERO] texto traducido
2. Traduce SOLO el texto, no los números
3. Preserva el tono, registro y puntuación
4. Subtítulos naturales y fluidos en {lang_name}
5. Responde SOLO con los subtítulos traducidos

SUBTÍTULOS:
{batch_text}"""

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=120
            )
            response.raise_for_status()
            translated_text = response.json().get("response", "")

            for line in translated_text.strip().split("\n"):
                match = re.match(r'\[(\d+)\]\s*(.*)', line.strip())
                if match:
                    idx = int(match.group(1))
                    text = match.group(2).strip()
                    for seg in batch:
                        if seg.index == idx:
                            seg.text = text
                            break

            translated_segments.extend(batch)

        logger.info("[Traducción] Traducción completada.")
        return translated_segments

    except Exception as e:
        logger.warning(f"[Traducción] Error: {e}")
        return segments


# ─────────────────────────────────────────────
# Descarga de audio desde YouTube
# ─────────────────────────────────────────────

def is_youtube_url(url: str) -> bool:
    """Detecta si una cadena es una URL de YouTube."""
    return (
        url.startswith("http://") or url.startswith("https://")
    ) and (
        "youtube.com" in url or "youtu.be" in url
    )


def download_youtube_audio(url: str, output_dir: str, progress_callback=None) -> tuple[str, str]:
    """
    Descarga el audio de un video de YouTube usando yt-dlp.
    Retorna (ruta_al_wav, titulo_del_video).
    Requiere yt-dlp y ffmpeg instalados.
    """
    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError("yt-dlp no está instalado. Instala con: pip install yt-dlp")

    output_template = os.path.join(output_dir, "yt_audio.%(ext)s")
    video_title = [None]

    class ProgressHook:
        def __call__(self, d):
            if d["status"] == "downloading" and progress_callback:
                total = d.get("total_bytes") or d.get("total_bytes_estimate", 0)
                downloaded = d.get("downloaded_bytes", 0)
                if total:
                    pct = int(downloaded / total * 100)
                    progress_callback(f"Descargando video de YouTube... {pct}%", pct // 10)

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }],
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [ProgressHook()],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_title[0] = info.get("title", "youtube_video")

    wav_path = os.path.join(output_dir, "yt_audio.wav")
    if not os.path.exists(wav_path):
        # yt-dlp puede generar otro nombre si el postprocesador tuvo problemas
        candidates = list(Path(output_dir).glob("yt_audio.*"))
        if candidates:
            wav_path = str(candidates[0])
        else:
            raise RuntimeError("No se encontró el archivo de audio descargado de YouTube.")

    logger.info(f"[YouTube] Audio descargado: {wav_path} | Título: {video_title[0]}")
    return wav_path, video_title[0]


# ─────────────────────────────────────────────
# Pipeline principal orquestador
# ─────────────────────────────────────────────

def run_pipeline(
    audio_path: str,
    language: str = "es",
    model_size: str = "medium",
    llm_model: str = "llama3",
    gguf_model_path: Optional[str] = None,
    enable_denoise: bool = True,
    enable_diarization: bool = False,
    enable_correction: bool = True,
    enable_translation: bool = False,
    target_language: Optional[str] = None,
    auto_translate_to_spanish: bool = True,
    hf_token: Optional[str] = None,
    progress_callback=None,
) -> PipelineResult:
    """
    Orquesta todo el pipeline de IA.

    LLM: llama.cpp tiene prioridad. Si no hay modelo GGUF disponible,
    se intenta Ollama como fallback automático.

    Args:
        gguf_model_path: Ruta explícita a un .gguf (opcional).
                         Si None, se busca automáticamente en ~/.subtitle_ai/models/
        llm_model:       Modelo Ollama a usar como fallback (ej. "llama3")
        auto_translate_to_spanish: Traduce automáticamente si el audio no está en español.
    """

    def progress(step, pct):
        logger.info(f"[Pipeline] {step} ({pct}%)")
        if progress_callback:
            progress_callback(step, pct)

    errors = []

    # ── Inicializar LLMEngine (llama.cpp → Ollama → Null) ────
    progress("Inicializando motor LLM...", 1)
    try:
        from .llm_engine import LLMEngine
        llm_engine = LLMEngine.auto_detect(
            gguf_model_path=gguf_model_path,
            ollama_model=llm_model,
        )
        progress(f"Motor LLM: {llm_engine.backend_name}", 2)
    except Exception as e:
        logger.warning(f"[Pipeline] Error inicializando LLM: {e}")
        llm_engine = None
        errors.append(f"LLM no disponible: {e}")

    with tempfile.TemporaryDirectory() as tmpdir:

        # ── Paso 0: Diagnóstico de calidad de audio ───
        progress("Analizando calidad del audio...", 3)
        quality_report = analyze_audio_quality(audio_path)

        for w in quality_report.warnings:
            logger.warning(f"[Diagnóstico] {w}")
            errors.append(f"Advertencia de audio: {w}")

        if model_size in ("tiny", "base") and quality_report.quality_label in ("ACEPTABLE", "DIFÍCIL"):
            logger.info(f"[Diagnóstico] Calidad '{quality_report.quality_label}' → subiendo modelo a '{quality_report.recommended_model}'")
            model_size = quality_report.recommended_model
            errors.append(f"Modelo auto-ajustado a '{model_size}' por calidad {quality_report.quality_label}.")

        if quality_report.recommended_denoise and not enable_denoise:
            enable_denoise = True

        beam_size = quality_report.recommended_beam_size
        progress(f"Calidad de audio: {quality_report.quality_label} | SNR: {quality_report.snr_db}dB", 5)

        # ── Paso 1: Preprocesamiento ──────────────────
        progress("Preprocesando audio...", 7)
        wav_path = os.path.join(tmpdir, "audio.wav")
        processed_path = preprocess_audio(audio_path, wav_path)

        if enable_denoise:
            denoise_strength = {
                "EXCELENTE": 0.5, "BUENA": 0.70,
                "ACEPTABLE": 0.82, "DIFÍCIL": 0.90,
            }.get(quality_report.quality_label, 0.75)
            progress(f"Reduciendo ruido (intensidad {denoise_strength:.0%})...", 14)
            processed_path = _denoise_with_strength(processed_path, denoise_strength)

        # ── Paso 2: Transcripción ─────────────────────
        # Liberar VRAM del LLM antes de cargar Whisper
        # (en 8GB VRAM no pueden coexistir ambos)
        if llm_engine and hasattr(llm_engine, 'unload'):
            progress("Liberando VRAM para Whisper...", 20)
            llm_engine.unload()

        progress(f"Transcribiendo con Whisper '{model_size}' (beam={beam_size})...", 22)
        try:
            segments, metadata = _transcribe_with_beam(processed_path, language, model_size, beam_size)
        except Exception as e:
            raise RuntimeError(f"Error crítico en transcripción: {e}")

        detected_lang = metadata["language"]
        progress(f"Transcripción completada — idioma detectado: {detected_lang.upper()}", 55)

        # ── Auto-traducción al español ────────────────
        if auto_translate_to_spanish and detected_lang in NON_SPANISH_LANGUAGES:
            lang_full = LANGUAGE_NAMES.get(detected_lang, detected_lang)
            logger.info(f"[Auto-traducción] {lang_full} → español")
            progress(f"Idioma: {lang_full} → activando traducción a español...", 57)
            enable_translation = True
            target_language = "es"
            if not enable_correction:
                enable_correction = True

        # ── Paso 3: Diarización ───────────────────────
        speakers_found = 0
        if enable_diarization:
            progress("Identificando hablantes...", 60)
            diarization = diarize(processed_path, hf_token)
            if diarization:
                segments = assign_speakers(segments, diarization)
                speakers_found = len(set(s.speaker for s in segments if s.speaker))
            else:
                errors.append("Diarización no disponible (requiere pyannote + HF_TOKEN)")

        # ── Paso 4: Corrección LLM ────────────────────
        corrected = False
        if enable_correction:
            backend_label = llm_engine.backend_name if llm_engine else "no disponible"
            progress(f"Corrigiendo con LLM ({backend_label})...", 70)
            original_texts = [s.text for s in segments]
            segments = correct_with_llm(segments, llm_engine)
            corrected = any(s.text != o for s, o in zip(segments, original_texts))
            if not corrected:
                errors.append(f"Corrección LLM no aplicada ({backend_label})")

        # ── Paso 5: Traducción ────────────────────────
        translated = False
        if enable_translation and target_language:
            if detected_lang == target_language:
                errors.append(f"Traducción omitida: el audio ya está en {LANGUAGE_NAMES.get(target_language, target_language)}.")
            else:
                lang_name = LANGUAGE_NAMES.get(target_language, target_language)
                progress(f"Traduciendo a {lang_name}...", 85)
                segments = translate_with_llm(segments, target_language, llm_engine)
                translated = True

        progress("Generando archivo .srt...", 95)

        return PipelineResult(
            segments=segments,
            language=detected_lang,
            language_probability=metadata["language_probability"],
            duration=metadata.get("duration", quality_report.duration_seconds),
            speakers_found=speakers_found,
            corrected=corrected,
            translated=translated,
            target_language=target_language,
            errors=errors,
            audio_quality=quality_report,
        )


# ─────────────────────────────────────────────
# Helpers internos del pipeline
# ─────────────────────────────────────────────

def _denoise_with_strength(audio_path: str, prop_decrease: float) -> str:
    """
    Wrapper de denoise_audio que permite ajustar la agresividad.
    Factoriza la lógica para no duplicar código.
    """
    try:
        import noisereduce as nr
        import librosa
        import soundfile as sf

        logger.info(f"[Denoise] prop_decrease={prop_decrease}")
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        audio_clean = nr.reduce_noise(
            y=audio,
            sr=sr,
            stationary=False,
            prop_decrease=prop_decrease,
            freq_mask_smooth_hz=500,
            time_mask_smooth_ms=50,
        )
        clean_path = audio_path.replace(".wav", "_clean.wav")
        sf.write(clean_path, audio_clean, sr)
        return clean_path
    except ImportError:
        logger.warning("[Denoise] noisereduce no instalado.")
        return audio_path
    except Exception as e:
        logger.warning(f"[Denoise] Error: {e}")
        return audio_path


def _transcribe_with_beam(
    audio_path: str,
    language: str,
    model_size: str,
    beam_size: int,
) -> tuple[list[Segment], dict]:
    """
    Wrapper de transcribe() que expone beam_size como parámetro.
    beam_size alto = más preciso pero más lento.
    Recomendado: 5 (normal), 8 (audio ruidoso), 10 (audio difícil).
    """
    from faster_whisper import WhisperModel

    logger.info(f"[ASR] Modelo={model_size} | beam_size={beam_size} | lang={language}")

    device, compute = "cpu", "int8"
    try:
        import torch
        if torch.cuda.is_available():
            device, compute = "cuda", "float16"
            logger.info("[ASR] GPU detectada → usando CUDA float16.")
    except ImportError:
        pass

    model = WhisperModel(model_size, device=device, compute_type=compute)

    raw_segments, info = model.transcribe(
        audio_path,
        language=language if language != "auto" else None,
        beam_size=beam_size,
        best_of=max(3, beam_size // 2),        # evalúa múltiples candidatos
        temperature=0.0,                        # determinista: más consistente
        condition_on_previous_text=True,        # usa contexto previo del audio
        vad_filter=True,
        vad_parameters={
            "threshold": 0.45,
            "min_speech_duration_ms": 200,
            "min_silence_duration_ms": 400,
            "speech_pad_ms": 200,
        },
        word_timestamps=True,
    )

    segments = [
        Segment(
            index=i,
            start=seg.start,
            end=seg.end,
            text=seg.text.strip(),
            confidence=getattr(seg, "avg_logprob", 1.0),
        )
        for i, seg in enumerate(raw_segments, 1)
    ]

    metadata = {
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "duration": getattr(info, "duration", 0),
    }

    logger.info(f"[ASR] {len(segments)} segmentos | lang={info.language} ({info.language_probability:.0%})")
    return segments, metadata
