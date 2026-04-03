# SubtitleAI

**Generador de subtítulos `.srt` con inteligencia artificial 100% local.**

Transcribe audio y video usando Whisper, corrige el texto con un LLM y traduce opcionalmente — sin enviar ningún dato a la nube.

---

## Características

- **Transcripción con Whisper** — modelos tiny → large-v3, detección automática de idioma
- **Reducción de ruido adaptativa** — análisis SNR, filtrado espectral, diagnóstico de calidad
- **Corrección con LLM** — ortografía, puntuación y nombres propios corregidos automáticamente
- **Traducción automática** — al español u otro idioma destino
- **Identificación de hablantes** — diarización con pyannote (opcional)
- **YouTube** — descarga, selección de formato y generación de `.srt` en un solo paso
- **Interfaz web y CLI** — drag-and-drop en el navegador o control total desde la terminal
- **100% local** — ningún dato sale del equipo

---

## Pipeline de procesamiento

```
Audio / Video / URL de YouTube
         │
         ▼
[0] Diagnóstico de calidad      librosa — SNR, ratio de voz, tipo de ruido
         │                      Ajusta modelo Whisper y beam_size automáticamente
         ▼
[1] Preprocesamiento            ffmpeg → WAV mono 16 kHz
         │                      noisereduce → reducción de ruido adaptativa
         ▼
[2] Transcripción ASR           faster-whisper (float16 en GPU)
         │                      VAD filter, word timestamps, beam search
         ▼
[3] Diarización (opcional)      pyannote/speaker-diarization-3.1
         │                      Asigna Hablante A / B / C a cada segmento
         ▼
[4] Corrección LLM              llama.cpp (principal) → Ollama (fallback)
         │                      Corrige ortografía, puntuación, nombres propios
         ▼
[5] Traducción LLM (opcional)   Mismo motor LLM
         │                      Auto-traduce al español si el audio no es en español
         ▼
      archivo .srt
```

---

## Hardware recomendado

| Componente | Mínimo | Configuración de desarrollo |
|---|---|---|
| GPU | — | NVIDIA RTX 5060 Laptop 8 GB (Blackwell sm_120) |
| RAM | 8 GB | 64 GB |
| Python | 3.10+ | 3.12.3 |
| SO | Ubuntu 20.04+ | Ubuntu 24.04 LTS |
| CUDA | 11.8+ | 12.8 |

> La GPU es opcional. Sin GPU el pipeline funciona en CPU (más lento).

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone git@github.com:jpalacioscl/subtitulos_srt.git
cd subtitulos_srt
```

### 2. Instalar dependencias y modelos

```bash
python3 setup_blackwell.py
```

El instalador crea automáticamente un entorno virtual (`venv/`) con:

- PyTorch 2.10 + CUDA 12.8
- faster-whisper
- llama-cpp-python compilado con CUDA
- librosa, noisereduce, soundfile
- Flask, yt-dlp, ffmpeg

Y descarga el modelo GGUF recomendado (~4.9 GB):

```
~/.subtitle_ai/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
```

### Opciones del instalador

```bash
python3 setup_blackwell.py             # instalación completa + modelo
python3 setup_blackwell.py --no-model  # solo dependencias, sin modelo
python3 setup_blackwell.py --download  # solo descargar el modelo GGUF
python3 setup_blackwell.py --verify    # verificar el entorno instalado
```

### Verificar instalación

```bash
source venv/bin/activate
python3 setup_blackwell.py --verify
```

Salida esperada:

```
  ✓ Python 3.12.3
  ✓ PyTorch 2.10.0+cu128 | CUDA 12.8
  ✓ GPU: NVIDIA GeForce RTX 5060 Laptop GPU | 7.5 GB VRAM | cc=12.0 [Blackwell ✓]
  ✓ faster-whisper instalado
  ✓ llama-cpp-python instalado con soporte GPU (CUDA)
  ✓ Modelos GGUF: 1 archivo (4.9 GB)
  ✓ Flask / yt-dlp / ffmpeg
  ✓ RAM: 63 GB
```

---

## Uso

### Interfaz web (recomendado)

```bash
./run_web.sh
```

Abre `http://localhost:5000`. Incluye arrastrar y soltar, selección de formato de YouTube, progreso en tiempo real y descarga directa del `.srt`.

### CLI

```bash
./run_cli.sh <archivo_o_url> [opciones]
```

#### Ejemplos rápidos

```bash
# Transcripción básica (idioma auto-detectado)
./run_cli.sh video.mp4

# Especificar idioma
./run_cli.sh audio.wav --language en

# Modelo Whisper más preciso
./run_cli.sh video.mp4 --model large-v2

# Traducir al español
./run_cli.sh video.mp4 --translate es

# YouTube: descargar, transcribir y generar .srt
./run_cli.sh "https://www.youtube.com/watch?v=..." --language es

# Identificar hablantes (requiere HF_TOKEN)
./run_cli.sh video.mp4 --diarize --hf-token TU_TOKEN

# Sin reducción de ruido ni corrección LLM
./run_cli.sh audio.wav --no-denoise --no-correct

# Usar un modelo GGUF específico
./run_cli.sh video.mp4 --gguf ~/.subtitle_ai/models/mistral.gguf
```

#### Todas las opciones

```
positional:
  input                Archivo de audio/video o URL de YouTube

Transcripción:
  -l, --language       Idioma del audio: auto, es, en, fr, de, it, pt... (default: auto)
  -m, --model          Modelo Whisper: tiny, base, small, medium, large-v2, large-v3

Procesamiento de audio:
  --no-denoise         Desactivar reducción de ruido
  --no-quality-check   Omitir diagnóstico de calidad

Motor LLM:
  --no-correct         Desactivar corrección LLM
  --translate LANG     Traducir subtítulos al idioma indicado
  --no-auto-translate  No traducir automáticamente al español
  --gguf PATH          Ruta a modelo GGUF personalizado
  --llm-model MODEL    Modelo Ollama como fallback (default: llama3)

Diarización:
  --diarize            Identificar hablantes (requiere pyannote y HF_TOKEN)
  --hf-token TOKEN     Token de HuggingFace

YouTube:
  --download-dir DIR   Carpeta de descarga (default: ~/Downloads)

Otros:
  -o, --output FILE    Archivo .srt de salida
  -v, --verbose        Mostrar logs detallados
```

---

## Motor LLM

El pipeline detecta y selecciona automáticamente el mejor backend disponible:

| Prioridad | Backend | Cuándo se activa |
|---|---|---|
| 1 | **llama.cpp** | Hay un `.gguf` en `~/.subtitle_ai/models/` |
| 2 | **Ollama** | Servidor corriendo en `localhost:11434` |
| 3 | **Null** | Sin LLM — el pipeline continúa sin corrección |

### Modelos GGUF soportados

| Modelo | Tamaño disco | VRAM aprox. | Calidad |
|---|---|---|---|
| Llama 3.1 8B Q4_K_M | 4.9 GB | ~5.5 GB | ⭐⭐⭐⭐ recomendado |
| Llama 3.1 8B Q5_K_M | 5.7 GB | ~6.2 GB | ⭐⭐⭐⭐⭐ |
| Mistral 7B Q4_K_M | 4.4 GB | ~5.0 GB | ⭐⭐⭐⭐ más rápido |
| Phi-3 Mini Q4_K_M | 2.2 GB | ~2.8 GB | ⭐⭐⭐ ultraligero |
| Llama 3.3 70B Q2_K | 26 GB | solo RAM | ⭐⭐⭐⭐⭐+ muy lento |

Coloca cualquier modelo `.gguf` compatible en `~/.subtitle_ai/models/` y se detectará automáticamente.

---

## Estructura del proyecto

```
subtitulos_srt/
├── core/
│   ├── pipeline.py          # orquestador del pipeline completo (5 etapas)
│   ├── llm_engine.py        # motor LLM: llama.cpp / Ollama / Null
│   └── model_manager.py     # descarga y gestión de modelos GGUF
├── app_flask.py             # interfaz web (Flask, drag-and-drop, YouTube)
├── subtitles_cli.py         # CLI con barra de progreso
├── setup_blackwell.py       # instalador automático para RTX Blackwell
├── run_cli.sh               # lanzador CLI (activa venv automáticamente)
└── run_web.sh               # lanzador web (activa venv automáticamente)
```

---

## Tecnologías principales

| Componente | Tecnología |
|---|---|
| Transcripción | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) |
| Inferencia LLM | [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) / [Ollama](https://ollama.com) |
| Reducción de ruido | [noisereduce](https://github.com/timsainb/noisereduce) + librosa |
| Diarización | [pyannote.audio](https://github.com/pyannote/pyannote-audio) |
| YouTube | [yt-dlp](https://github.com/yt-dlp/yt-dlp) |
| Interfaz web | Flask |
| Deep learning | PyTorch + CUDA |

---

## Licencia

MIT
