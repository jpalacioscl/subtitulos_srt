# SubtitleAI

Generador de subtítulos `.srt` con IA local. Transcribe audio/video usando Whisper, corrige con un LLM y opcionalmente traduce, todo sin enviar datos a la nube.

## Hardware recomendado

| Componente | Mínimo | Este proyecto |
|---|---|---|
| GPU | — | RTX 5060 Laptop (8 GB VRAM, Blackwell sm_120) |
| RAM | 8 GB | 64 GB |
| Python | 3.10+ | 3.12.3 |
| OS | Ubuntu | Ubuntu 24.04 LTS |

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone git@github.com:jpalacioscl/subtitulos_srt.git
cd subtitulos_srt
```

### 2. Instalar dependencias

```bash
python3 setup_blackwell.py
```

Esto instala automáticamente en un entorno virtual (`venv/`):

- PyTorch 2.10 + CUDA 12.8 (Blackwell)
- faster-whisper
- llama-cpp-python compilado con CUDA
- librosa, noisereduce, soundfile
- Flask, FastAPI, uvicorn

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

---

## Uso

### App web (recomendado)

```bash
./run_web.sh
```

Abre `http://localhost:5000` en el navegador. Interfaz de arrastrar y soltar, con progreso en tiempo real y descarga directa del `.srt`.

### CLI (terminal)

```bash
./run_cli.sh video.mp4
```

#### Ejemplos

```bash
# Transcripción básica (idioma auto-detectado)
./run_cli.sh video.mp4

# Especificar idioma del audio
./run_cli.sh audio.wav --language en

# Usar modelo Whisper más preciso
./run_cli.sh video.mp4 --model large-v2

# Traducir al español automáticamente
./run_cli.sh video.mp4 --translate es

# Guardar en un archivo específico
./run_cli.sh video.mp4 --output mis_subtitulos.srt

# Identificar hablantes (requiere HF_TOKEN)
./run_cli.sh video.mp4 --diarize --hf-token TU_TOKEN

# Sin reducción de ruido ni corrección LLM
./run_cli.sh audio.wav --no-denoise --no-correct

# Usar un modelo GGUF específico
./run_cli.sh video.mp4 --gguf ~/.subtitle_ai/models/mistral.gguf
```

#### Todas las opciones

```
positional arguments:
  input                Archivo de audio o video (mp4, mkv, mp3, wav, m4a...)

Transcripción (Whisper):
  -l, --language       Idioma del audio: es, en, fr, de, auto... (default: auto)
  -m, --model          Modelo Whisper: tiny, base, small, medium, large-v2, large-v3

Procesamiento de audio:
  --no-denoise         Desactivar reducción de ruido
  --no-quality-check   Omitir diagnóstico de calidad

Motor LLM:
  --no-correct         Desactivar corrección LLM
  --translate LANG     Traducir subtítulos al idioma indicado (es, en, fr...)
  --no-auto-translate  No traducir automáticamente al español
  --gguf PATH          Ruta a modelo GGUF para llama.cpp
  --llm-model MODEL    Modelo Ollama como fallback (default: llama3)

Diarización:
  --diarize            Identificar hablantes (requiere HF_TOKEN y pyannote)
  --hf-token TOKEN     Token de HuggingFace para pyannote

Otros:
  -o, --output         Archivo .srt de salida
  -v, --verbose        Mostrar logs detallados
```

---

## Pipeline de procesamiento

```
Audio/Video
    │
    ▼
[0] Diagnóstico de calidad         librosa — SNR, ratio de voz, tipo de ruido
    │                              Ajusta modelo Whisper y beam_size automáticamente
    ▼
[1] Preprocesamiento               ffmpeg → WAV mono 16kHz
    │                              noisereduce → reducción de ruido adaptativa
    ▼
[2] Transcripción ASR              faster-whisper (float16 en GPU Blackwell)
    │                              VAD filter, word timestamps, beam search
    ▼
[3] Diarización (opcional)         pyannote/speaker-diarization-3.1
    │                              Asigna Hablante A/B/C a cada segmento
    ▼
[4] Corrección LLM                 llama.cpp (principal) → Ollama (fallback)
    │                              Corrige ortografía, puntuación, nombres propios
    ▼
[5] Traducción LLM (opcional)      Mismo motor LLM
    │                              Auto-traduce al español si el audio no es en español
    ▼
Archivo .srt
```

---

## Motor LLM

El pipeline detecta automáticamente el mejor backend disponible:

| Prioridad | Backend | Cuándo se usa |
|---|---|---|
| 1 | **llama.cpp** | Hay un `.gguf` en `~/.subtitle_ai/models/` |
| 2 | **Ollama** | Servidor Ollama corriendo en `localhost:11434` |
| 3 | **Null** | Sin LLM — el pipeline continúa sin corrección |

### Modelos GGUF disponibles

| Modelo | Tamaño | VRAM | Calidad |
|---|---|---|---|
| Llama 3.1 8B Q4_K_M | 4.9 GB | ~5.5 GB | ⭐⭐⭐⭐ recomendado |
| Llama 3.1 8B Q5_K_M | 5.7 GB | ~6.2 GB | ⭐⭐⭐⭐⭐ |
| Mistral 7B Q4_K_M | 4.4 GB | ~5.0 GB | ⭐⭐⭐⭐ más rápido |
| Phi-3 Mini Q4_K_M | 2.2 GB | ~2.8 GB | ⭐⭐⭐ ligero |
| Llama 3.3 70B Q2_K | 26 GB | solo RAM | ⭐⭐⭐⭐⭐+ lento |

Para descargar un modelo diferente al recomendado, colócalo en `~/.subtitle_ai/models/` y se detectará automáticamente.

---

## Estructura del proyecto

```
subtitulos_srt/
├── core/
│   ├── __init__.py
│   ├── pipeline.py          # orquestador del pipeline completo
│   ├── llm_engine.py        # motor LLM: llama.cpp / Ollama / Null
│   └── model_manager.py     # descarga y gestión de modelos GGUF
├── app_flask.py             # app web con interfaz de subida y descarga
├── subtitles_cli.py         # CLI con barra de progreso
├── setup_blackwell.py       # instalador para RTX 5060 Blackwell
├── run_cli.sh               # lanzador CLI (activa venv automáticamente)
├── run_web.sh               # lanzador web (activa venv automáticamente)
└── venv/                    # entorno virtual Python (no se sube a git)
```

---

## Verificar instalación

```bash
source venv/bin/activate
python3 setup_blackwell.py --verify
```

Salida esperada:

```
  ✓ Python 3.12.3
  ✓ PyTorch 2.10.0+cu128 | CUDA 12.8
  ✓ GPU: NVIDIA GeForce RTX 5060 Laptop GPU | 7.5GB VRAM | cc=12.0 [Blackwell ✓]
  ✓ faster-whisper instalado
  ✓ llama-cpp-python instalado
  ✓ llama.cpp con soporte GPU (CUDA)
  ✓ Modelos GGUF: 1 archivos (4.9 GB total)
  ✓ Flask / FastAPI / Uvicorn
  ✓ ffmpeg
  ✓ RAM: 63 GB
```
