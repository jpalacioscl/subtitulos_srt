#!/usr/bin/env python3
"""
setup_blackwell.py
==================
Script de instalación completo para:
  - Ubuntu 24.04.4 LTS
  - Intel Core 7 240H
  - NVIDIA GeForce RTX 5060 Laptop (Blackwell, sm_120)
  - 64 GB RAM

Instala y configura:
  1. Dependencias del sistema (ffmpeg, build tools)
  2. PyTorch con soporte CUDA 12.8 (Blackwell)
  3. faster-whisper (float16 para Blackwell)
  4. llama-cpp-python con CUDA
  5. Dependencias del pipeline (librosa, noisereduce, flask, fastapi)
  6. huggingface-hub para descargar modelos GGUF
  7. Descarga el modelo GGUF recomendado

Uso:
    python setup_blackwell.py              # instalación completa
    python setup_blackwell.py --verify     # solo verificar entorno
    python setup_blackwell.py --download   # solo descargar modelo GGUF
"""

import sys
import subprocess
import os
import argparse
from pathlib import Path

# ─────────────────────────────────────────────
# Colores ANSI
# ─────────────────────────────────────────────
G  = "\033[92m"   # verde
Y  = "\033[93m"   # amarillo
R  = "\033[91m"   # rojo
C  = "\033[96m"   # cyan
B  = "\033[1m"    # bold
RS = "\033[0m"    # reset

def ok(msg):    print(f"  {G}✓{RS} {msg}")
def warn(msg):  print(f"  {Y}⚠{RS} {msg}")
def err(msg):   print(f"  {R}✗{RS} {msg}", file=sys.stderr)
def info(msg):  print(f"  {C}→{RS} {msg}")
def section(t): print(f"\n{B}{C}{'─'*55}\n  {t}\n{'─'*55}{RS}")

def run(cmd, check=True, capture=False):
    """Ejecuta un comando y retorna el resultado."""
    kwargs = {"check": check}
    if capture:
        kwargs["capture_output"] = True
        kwargs["text"] = True
    result = subprocess.run(cmd, shell=isinstance(cmd, str), **kwargs)
    return result


# ─────────────────────────────────────────────
# 1 — Verificar Python
# ─────────────────────────────────────────────

def check_python():
    section("Verificando Python")
    v = sys.version_info
    if v >= (3, 10):
        ok(f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        err(f"Python {v.major}.{v.minor} — Se requiere 3.10+")
        sys.exit(1)


# ─────────────────────────────────────────────
# 2 — Dependencias del sistema
# ─────────────────────────────────────────────

def install_system_deps():
    section("Instalando dependencias del sistema")
    info("sudo apt-get update + ffmpeg + build-essential + cmake")

    try:
        run("sudo apt-get update -qq", check=False)
        run(
            "sudo apt-get install -y ffmpeg build-essential cmake "
            "python3-dev libsndfile1 git curl wget libgomp1",
            check=False
        )
        # Verificar ffmpeg
        r = run("ffmpeg -version", check=False, capture=True)
        if r.returncode == 0:
            ok("ffmpeg instalado")
        else:
            warn("ffmpeg no instalado — algunas funciones no estarán disponibles")
    except Exception as e:
        warn(f"No se pudieron instalar dependencias del sistema: {e}")


# ─────────────────────────────────────────────
# 3 — PyTorch con CUDA 12.8 (Blackwell)
# ─────────────────────────────────────────────

def install_pytorch():
    section("Instalando PyTorch con soporte CUDA 12.8 (Blackwell)")
    info("RTX 5060 requiere CUDA 12.8+ para compute capability sm_120")

    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            cc = f"{props.major}.{props.minor}"
            ok(f"PyTorch {torch.__version__} ya instalado — GPU: {props.name} (cc={cc})")
            if props.major >= 10:
                ok("Blackwell detectado y soportado")
            return
        else:
            warn("PyTorch instalado pero sin CUDA — reinstalando con CUDA 12.8")
    except ImportError:
        info("PyTorch no instalado — instalando...")

    run(
        f"{sys.executable} -m pip install torch torchvision torchaudio "
        "--index-url https://download.pytorch.org/whl/cu128 --upgrade"
    )
    ok("PyTorch con CUDA 12.8 instalado")


# ─────────────────────────────────────────────
# 4 — faster-whisper + CUDA libs
# ─────────────────────────────────────────────

def install_faster_whisper():
    section("Instalando faster-whisper")
    info("Usando float16 para Blackwell (int8 no compatible con sm_120)")

    run(f"{sys.executable} -m pip install faster-whisper --upgrade")
    run(f"{sys.executable} -m pip install nvidia-cublas-cu12 'nvidia-cudnn-cu12==9.*' --upgrade")

    # Configurar LD_LIBRARY_PATH permanentemente
    bashrc = Path.home() / ".bashrc"
    ld_line = (
        'export LD_LIBRARY_PATH=$(python3 -c '
        "'import os,nvidia.cublas.lib,nvidia.cudnn.lib;"
        "print(os.path.dirname(nvidia.cublas.lib.__file__)+':'"
        "+os.path.dirname(nvidia.cudnn.lib.__file__))')"
    )

    current = bashrc.read_text() if bashrc.exists() else ""
    if "nvidia.cublas.lib" not in current:
        with open(bashrc, "a") as f:
            f.write(f"\n# SubtitleAI — CUDA libs para faster-whisper\n{ld_line}\n")
        ok(f"LD_LIBRARY_PATH añadido a ~/.bashrc")
        warn("Ejecuta: source ~/.bashrc  (o abre una nueva terminal)")
    else:
        ok("LD_LIBRARY_PATH ya configurado en ~/.bashrc")

    ok("faster-whisper instalado")


# ─────────────────────────────────────────────
# 5 — llama-cpp-python con CUDA
# ─────────────────────────────────────────────

def install_llama_cpp():
    section("Instalando llama-cpp-python con CUDA (Blackwell)")
    info("Compilando con CMAKE_ARGS=-DGGML_CUDA=on")
    info("Esto puede tardar 3-5 minutos la primera vez...")

    # Verificar si ya está instalado con CUDA
    try:
        from llama_cpp import Llama
        # Verificar que tiene soporte CUDA
        r = run(
            f"{sys.executable} -c \"from llama_cpp import llama_cpp; "
            "print(llama_cpp.llama_supports_gpu_offload())\"",
            check=False, capture=True
        )
        if "True" in (r.stdout or ""):
            ok("llama-cpp-python ya instalado con soporte CUDA")
            return
        else:
            warn("llama-cpp-python sin CUDA — reinstalando...")
    except ImportError:
        info("llama-cpp-python no instalado — compilando con CUDA...")

    env = os.environ.copy()
    env["CMAKE_ARGS"] = "-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=120"
    env["FORCE_CMAKE"] = "1"

    cmd = [
        sys.executable, "-m", "pip", "install",
        "llama-cpp-python", "--force-reinstall", "--no-cache-dir",
        "--upgrade"
    ]

    try:
        result = subprocess.run(cmd, env=env, check=True)
        ok("llama-cpp-python compilado con CUDA para Blackwell (sm_120)")
    except subprocess.CalledProcessError:
        warn("Compilación con CUDA falló — instalando versión CPU como fallback")
        run(f"{sys.executable} -m pip install llama-cpp-python --upgrade")
        warn("llama.cpp correrá en CPU. Para CUDA: CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python --force-reinstall")


# ─────────────────────────────────────────────
# 6 — Resto de dependencias del pipeline
# ─────────────────────────────────────────────

def install_pipeline_deps():
    section("Instalando dependencias del pipeline")

    packages = [
        ("flask>=3.0.0",           "Flask (app web)"),
        ("fastapi>=0.110.0",       "FastAPI (API REST)"),
        ("uvicorn[standard]",      "Uvicorn (servidor ASGI)"),
        ("python-multipart",       "Multipart (subida de archivos)"),
        ("websockets>=12.0",       "WebSockets (progreso en tiempo real)"),
        ("librosa>=0.10.0",        "librosa (análisis de audio)"),
        ("soundfile>=0.12.1",      "soundfile (lectura/escritura WAV)"),
        ("noisereduce>=3.0.0",     "noisereduce (reducción de ruido)"),
        ("psutil>=5.9.0",          "psutil (info de hardware)"),
        ("requests>=2.31.0",       "requests (cliente HTTP)"),
        ("huggingface-hub>=0.20.0","huggingface-hub (descargar modelos)"),
        ("numpy>=1.24.0",          "numpy"),
        ("tqdm>=4.65.0",           "tqdm (barras de progreso)"),
    ]

    for pkg, label in packages:
        try:
            run(f"{sys.executable} -m pip install '{pkg}' -q", check=False)
            ok(label)
        except Exception as e:
            warn(f"{label}: {e}")


# ─────────────────────────────────────────────
# 7 — Descargar modelo GGUF recomendado
# ─────────────────────────────────────────────

def download_recommended_model():
    section("Descargando modelo GGUF recomendado")
    info("Llama 3.1 8B Instruct Q4_K_M — 4.9 GB")
    info("Óptimo para RTX 5060: 5.5 GB VRAM, 40+ tok/s")

    models_dir = Path.home() / ".subtitle_ai" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    dest = models_dir / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

    if dest.exists():
        ok(f"Modelo ya descargado: {dest}")
        return str(dest)

    try:
        from huggingface_hub import hf_hub_download
        info("Descargando desde HuggingFace Hub...")
        info("(puede tardar varios minutos según la conexión)")

        path = hf_hub_download(
            repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            filename="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            local_dir=str(models_dir),
            local_dir_use_symlinks=False,
        )
        ok(f"Modelo descargado: {path}")
        return path

    except Exception as e:
        warn(f"No se pudo descargar automáticamente: {e}")
        info("Descarga manual:")
        info("  https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF")
        info(f"  Guardar en: {models_dir}/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
        return None


# ─────────────────────────────────────────────
# 8 — Verificación final del entorno
# ─────────────────────────────────────────────

def verify_environment():
    section("Verificación del entorno")

    all_ok = True

    # Python
    v = sys.version_info
    ok(f"Python {v.major}.{v.minor}.{v.micro}")

    # PyTorch + CUDA
    try:
        import torch
        cuda = torch.cuda.is_available()
        if cuda:
            props = torch.cuda.get_device_properties(0)
            vram = props.total_memory / 1024**3
            cc = f"{props.major}.{props.minor}"
            is_blackwell = props.major >= 10
            ok(f"PyTorch {torch.__version__} | CUDA {torch.version.cuda}")
            ok(f"GPU: {props.name} | {vram:.1f}GB VRAM | cc={cc} {'[Blackwell ✓]' if is_blackwell else ''}")

            if is_blackwell:
                ok("compute_type=float16 configurado automáticamente para Blackwell")
        else:
            warn("PyTorch sin CUDA — modo CPU")
    except ImportError:
        err("PyTorch no instalado")
        all_ok = False

    # faster-whisper
    try:
        from faster_whisper import WhisperModel
        ok("faster-whisper instalado")
    except ImportError:
        err("faster-whisper no instalado")
        all_ok = False

    # llama-cpp-python
    try:
        from llama_cpp import Llama
        ok("llama-cpp-python instalado")

        # Verificar soporte GPU
        try:
            from llama_cpp import llama_cpp
            gpu_support = llama_cpp.llama_supports_gpu_offload()
            if gpu_support:
                ok("llama.cpp con soporte GPU (CUDA)")
            else:
                warn("llama.cpp sin soporte GPU — modo CPU. Re-instala con: CMAKE_ARGS='-DGGML_CUDA=on'")
        except Exception:
            info("No se pudo verificar soporte GPU de llama.cpp")
    except ImportError:
        err("llama-cpp-python no instalado")
        all_ok = False

    # Modelo GGUF
    models_dir = Path.home() / ".subtitle_ai" / "models"
    gguf_files = list(models_dir.glob("*.gguf")) if models_dir.exists() else []
    if gguf_files:
        total_gb = sum(f.stat().st_size for f in gguf_files) / 1024**3
        ok(f"Modelos GGUF: {len(gguf_files)} archivos ({total_gb:.1f} GB total)")
        for f in gguf_files:
            info(f"  {f.name} ({f.stat().st_size/1024**3:.1f} GB)")
    else:
        warn(f"Sin modelos GGUF en {models_dir}")
        warn("Ejecuta: python setup_blackwell.py --download")

    # Flask / FastAPI
    for pkg, name in [("flask", "Flask"), ("fastapi", "FastAPI"), ("uvicorn", "Uvicorn")]:
        try:
            __import__(pkg)
            ok(name)
        except ImportError:
            err(f"{name} no instalado")
            all_ok = False

    # ffmpeg
    r = subprocess.run("ffmpeg -version", shell=True, capture_output=True)
    if r.returncode == 0:
        ok("ffmpeg")
    else:
        err("ffmpeg no instalado — sudo apt install ffmpeg")
        all_ok = False

    # RAM
    try:
        import psutil
        ram = psutil.virtual_memory().total / 1024**3
        ok(f"RAM: {ram:.0f} GB")
    except ImportError:
        pass

    print()
    if all_ok:
        print(f"  {G}{B}✓ Entorno listo. Ejecuta: python app_flask.py{RS}")
    else:
        print(f"  {Y}{B}⚠ Algunos componentes requieren atención (ver arriba){RS}")

    return all_ok


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Setup SubtitleAI para Alienware RTX 5060")
    parser.add_argument("--verify",   action="store_true", help="Solo verificar el entorno")
    parser.add_argument("--download", action="store_true", help="Solo descargar modelo GGUF")
    parser.add_argument("--no-model", action="store_true", help="No descargar modelo GGUF")
    args = parser.parse_args()

    print(f"""
{C}{B}
  ╔══════════════════════════════════════════════════════╗
  ║         SubtitleAI — Setup Alienware RTX 5060        ║
  ║         Ubuntu 24.04 · Blackwell · 64GB RAM          ║
  ╚══════════════════════════════════════════════════════╝
{RS}""")

    if args.verify:
        verify_environment()
        return

    if args.download:
        download_recommended_model()
        return

    # Instalación completa
    check_python()
    install_system_deps()
    install_pytorch()
    install_faster_whisper()
    install_llama_cpp()
    install_pipeline_deps()

    if not args.no_model:
        download_recommended_model()

    verify_environment()

    print(f"""
{G}{B}
  ╔══════════════════════════════════════════════════════╗
  ║                  ¡Instalación completa!              ║
  ╚══════════════════════════════════════════════════════╝
{RS}
  {C}Próximos pasos:{RS}
    source ~/.bashrc              # cargar LD_LIBRARY_PATH
    python app_flask.py           # Opción 1: App web Flask
    python subtitles_cli.py ...   # Opción 2: Terminal
    python app_fastapi.py         # Opción 3: FastAPI + React

  {C}Verificar en cualquier momento:{RS}
    python setup_blackwell.py --verify
""")


if __name__ == "__main__":
    main()
