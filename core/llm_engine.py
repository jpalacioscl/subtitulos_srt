"""
core/llm_engine.py
==================
Motor LLM unificado con soporte para llama.cpp y Ollama como fallback.

Arquitectura:
  LLMEngine (clase principal)
    ├── Backend: LlamaCppBackend   ← llama-cpp-python, 100% local, sin servidor
    ├── Backend: OllamaBackend     ← Ollama REST API (fallback)
    └── Backend: NullBackend       ← no-op cuando nada está disponible

Por qué llama.cpp directamente (sin Ollama):
  - Sin proceso servidor separado — el modelo carga en el mismo proceso Python
  - Acceso completo a parámetros: temperature, top_p, repeat_penalty, mirostat
  - Soporte nativo CUDA/Metal/Vulkan sin capas intermedias
  - Para Blackwell (RTX 5060): usa CUDA offloading con n_gpu_layers=-1
  - Menos latencia por request (sin HTTP overhead)
  - Control total del contexto y la memoria

Modelos GGUF recomendados para Alienware RTX 5060 (8GB VRAM + 64GB RAM):
  - Llama-3.1-8B-Instruct-Q4_K_M.gguf     (4.9 GB) ← recomendado principal
  - Llama-3.1-8B-Instruct-Q5_K_M.gguf     (5.7 GB) ← mejor calidad
  - Mistral-7B-Instruct-v0.3-Q4_K_M.gguf  (4.4 GB) ← alternativa rápida
  - Phi-3-mini-4k-instruct-Q4_K_M.gguf    (2.2 GB) ← ultra ligero
  - Llama-3.3-70B-Q2_K.gguf               (26 GB)  ← en RAM, sin GPU
"""

import os
import re
import logging
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Directorio de modelos GGUF
# ─────────────────────────────────────────────

DEFAULT_MODELS_DIR = Path.home() / ".subtitle_ai" / "models"
DEFAULT_MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# Detección de hardware para llama.cpp
# ─────────────────────────────────────────────

def detect_llama_cpp_config() -> dict:
    """
    Detecta el hardware disponible y devuelve la configuración
    óptima para llama-cpp-python.

    Para RTX 5060 Blackwell con 8GB VRAM:
    - n_gpu_layers=-1 (offload todo a GPU)
    - n_ctx=4096 (contexto para subtítulos, suficiente)
    - n_threads = núcleos físicos CPU

    Para modo CPU puro (64GB RAM):
    - n_gpu_layers=0
    - n_threads máximo
    """
    import multiprocessing
    cpu_threads = multiprocessing.cpu_count()

    config = {
        "n_gpu_layers": 0,
        "n_ctx": 4096,
        "n_threads": max(4, cpu_threads - 2),
        "n_batch": 512,
        "use_mmap": True,
        "use_mlock": False,
        "verbose": False,
        "backend": "cpu",
    }

    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / 1024**3
            is_blackwell = props.major >= 10

            logger.info(f"[LlamaCPP] GPU: {props.name} | {vram_gb:.1f}GB VRAM | "
                        f"{'Blackwell' if is_blackwell else 'Arquitectura anterior'}")

            # Con 8GB VRAM: offload completo, el modelo 8B Q4_K_M ocupa ~5GB
            config["n_gpu_layers"] = -1       # -1 = offload todas las capas
            config["n_ctx"] = 4096
            config["n_batch"] = 512
            config["backend"] = "cuda"

            # Blackwell: usar tensor_split si hay múltiples GPUs (no aplica aquí)
            # pero sí ajustar el rope_freq_base para mejor precisión
            if is_blackwell:
                logger.info("[LlamaCPP] Blackwell detectado — configuración CUDA optimizada")
                config["rope_freq_base"] = 500000.0   # mejor para modelos modernos

    except ImportError:
        logger.info("[LlamaCPP] PyTorch no disponible — modo CPU")

    return config


# ─────────────────────────────────────────────
# Backends abstractos
# ─────────────────────────────────────────────

class LLMBackend(ABC):
    """Interfaz común para todos los backends de LLM."""

    @abstractmethod
    def is_available(self) -> bool:
        """Verifica si el backend está disponible y funcional."""
        ...

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.1) -> str:
        """Genera texto a partir de un prompt."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Nombre del backend para logging."""
        ...


# ─────────────────────────────────────────────
# Backend 1: llama-cpp-python (principal)
# ─────────────────────────────────────────────

class LlamaCppBackend(LLMBackend):
    """
    Backend directo con llama-cpp-python.
    Carga el modelo GGUF en el mismo proceso Python.
    Sin servidor externo, sin HTTP, máximo control.

    Instalación con soporte CUDA para Blackwell:
        CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall

    Instalación CPU-only:
        pip install llama-cpp-python
    """

    def __init__(self, model_path: str, **kwargs):
        self._model_path = model_path
        self._kwargs = kwargs
        self._llm = None
        self._lock = threading.Lock()  # thread-safe para Flask/FastAPI
        self._loaded = False

    def _load(self):
        """Carga el modelo en memoria (lazy loading)."""
        if self._loaded:
            return

        with self._lock:
            if self._loaded:
                return

            try:
                from llama_cpp import Llama

                hw_config = detect_llama_cpp_config()
                merged = {**hw_config, **self._kwargs}   # kwargs del usuario tienen prioridad

                logger.info(f"[LlamaCPP] Cargando modelo: {Path(self._model_path).name}")
                logger.info(f"[LlamaCPP] Config: n_gpu_layers={merged.get('n_gpu_layers')} | "
                            f"n_ctx={merged.get('n_ctx')} | backend={merged.get('backend')}")

                self._llm = Llama(
                    model_path=self._model_path,
                    n_gpu_layers=merged.get("n_gpu_layers", 0),
                    n_ctx=merged.get("n_ctx", 4096),
                    n_threads=merged.get("n_threads", 8),
                    n_batch=merged.get("n_batch", 512),
                    use_mmap=merged.get("use_mmap", True),
                    use_mlock=merged.get("use_mlock", False),
                    verbose=merged.get("verbose", False),
                    rope_freq_base=merged.get("rope_freq_base", 0),  # 0 = default del modelo
                )
                self._loaded = True
                logger.info(f"[LlamaCPP] ✓ Modelo cargado exitosamente")

            except ImportError:
                raise RuntimeError(
                    "llama-cpp-python no instalado.\n"
                    "Instala con CUDA (recomendado para RTX 5060):\n"
                    "  CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install llama-cpp-python --force-reinstall\n"
                    "O sin GPU:\n"
                    "  pip install llama-cpp-python"
                )
            except Exception as e:
                raise RuntimeError(f"Error cargando modelo GGUF: {e}")

    def is_available(self) -> bool:
        try:
            from llama_cpp import Llama
            return Path(self._model_path).exists()
        except ImportError:
            return False

    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.1) -> str:
        self._load()

        with self._lock:
            output = self._llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                top_k=40,
                repeat_penalty=1.1,
                stop=["</s>", "[INST]", "[/INST]", "###", "\n\n\n"],
                echo=False,
            )
            return output["choices"][0]["text"].strip()

    def generate_chat(self, system: str, user: str,
                      max_tokens: int = 1024, temperature: float = 0.1) -> str:
        """
        Genera usando el formato chat (más natural para instrucciones).
        Usa la API de chat completion de llama-cpp-python.
        """
        self._load()

        with self._lock:
            output = self._llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                repeat_penalty=1.1,
                stop=["</s>"],
            )
            return output["choices"][0]["message"]["content"].strip()

    def unload(self):
        """Libera la VRAM/RAM del modelo."""
        with self._lock:
            if self._llm is not None:
                del self._llm
                self._llm = None
                self._loaded = False
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                logger.info("[LlamaCPP] Modelo descargado de memoria.")

    @property
    def name(self) -> str:
        return f"llama.cpp ({Path(self._model_path).name})"


# ─────────────────────────────────────────────
# Backend 2: Ollama (fallback)
# ─────────────────────────────────────────────

class OllamaBackend(LLMBackend):
    """
    Backend Ollama via REST API.
    Fallback cuando llama.cpp no está disponible.
    """

    def __init__(self, model: str = "llama3", host: str = "http://localhost:11434"):
        self._model = model
        self._host = host

    def is_available(self) -> bool:
        try:
            import requests
            r = requests.get(f"{self._host}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.1) -> str:
        import requests
        response = requests.post(
            f"{self._host}/api/generate",
            json={
                "model": self._model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
            timeout=180,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()

    def generate_chat(self, system: str, user: str,
                      max_tokens: int = 1024, temperature: float = 0.1) -> str:
        import requests
        response = requests.post(
            f"{self._host}/api/chat",
            json={
                "model": self._model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
            timeout=180,
        )
        response.raise_for_status()
        return response.json()["message"]["content"].strip()

    @property
    def name(self) -> str:
        return f"Ollama ({self._model})"


# ─────────────────────────────────────────────
# Backend 3: Null (no-op)
# ─────────────────────────────────────────────

class NullBackend(LLMBackend):
    """No hace nada — el pipeline sigue sin LLM."""

    def is_available(self) -> bool:
        return False

    def generate(self, prompt: str, **kwargs) -> str:
        return ""

    def generate_chat(self, system: str, user: str, **kwargs) -> str:
        return ""

    @property
    def name(self) -> str:
        return "NullBackend (sin LLM)"


# ─────────────────────────────────────────────
# LLMEngine — orquestador principal
# ─────────────────────────────────────────────

class LLMEngine:
    """
    Motor LLM unificado. Selecciona automáticamente el mejor backend
    disponible en este orden de prioridad:

      1. llama.cpp (si hay modelo GGUF descargado)
      2. Ollama    (si el servidor está corriendo)
      3. Null      (pipeline continúa sin corrección/traducción)

    Uso típico:
        engine = LLMEngine.auto_detect()
        result = engine.correct_subtitles(segments)
        result = engine.translate_subtitles(segments, "es")
    """

    def __init__(self, backend: LLMBackend):
        self._backend = backend
        logger.info(f"[LLMEngine] Backend activo: {backend.name}")

    @classmethod
    def from_gguf(cls, model_path: str, **kwargs) -> "LLMEngine":
        """Crea engine con llama.cpp desde un archivo GGUF específico."""
        backend = LlamaCppBackend(model_path, **kwargs)
        if not backend.is_available():
            raise FileNotFoundError(f"Modelo GGUF no encontrado: {model_path}")
        return cls(backend)

    @classmethod
    def from_ollama(cls, model: str = "llama3") -> "LLMEngine":
        """Crea engine con Ollama."""
        backend = OllamaBackend(model)
        return cls(backend)

    @classmethod
    def auto_detect(cls,
                    gguf_model_path: Optional[str] = None,
                    ollama_model: str = "llama3",
                    models_dir: Optional[str] = None) -> "LLMEngine":
        """
        Detección automática del mejor backend disponible.

        Prioridad:
          1. GGUF explícito si se provee ruta
          2. Primer modelo GGUF encontrado en models_dir
          3. Ollama si está corriendo
          4. NullBackend

        Args:
            gguf_model_path: Ruta explícita a un .gguf
            ollama_model:    Nombre del modelo Ollama a usar como fallback
            models_dir:      Directorio donde buscar archivos .gguf
        """
        search_dir = Path(models_dir) if models_dir else DEFAULT_MODELS_DIR

        # ── Intentar llama.cpp ────────────────────────
        candidate_path = None

        if gguf_model_path and Path(gguf_model_path).exists():
            candidate_path = gguf_model_path
            logger.info(f"[LLMEngine] Usando GGUF explícito: {candidate_path}")
        elif search_dir.exists():
            # Buscar el modelo más grande disponible (mejor calidad)
            # Preferencia: llama3 > mistral > phi > cualquier otro
            gguf_files = sorted(search_dir.glob("*.gguf"), key=lambda p: p.stat().st_size, reverse=True)
            if gguf_files:
                # Priorizar por nombre
                priority = ["llama-3", "llama3", "mistral", "qwen", "phi", "gemma"]
                for pref in priority:
                    match = next((f for f in gguf_files if pref in f.name.lower()), None)
                    if match:
                        candidate_path = str(match)
                        break
                if not candidate_path:
                    candidate_path = str(gguf_files[0])   # el más grande

        if candidate_path:
            try:
                backend = LlamaCppBackend(candidate_path)
                if backend.is_available():
                    logger.info(f"[LLMEngine] ✓ llama.cpp seleccionado: {Path(candidate_path).name}")
                    return cls(backend)
            except Exception as e:
                logger.warning(f"[LLMEngine] llama.cpp no disponible: {e}")

        # ── Fallback: Ollama ──────────────────────────
        ollama = OllamaBackend(ollama_model)
        if ollama.is_available():
            logger.info(f"[LLMEngine] ✓ Ollama seleccionado como fallback: {ollama_model}")
            return cls(ollama)

        # ── Fallback final: Null ──────────────────────
        logger.warning("[LLMEngine] ⚠ Ningún backend LLM disponible. Corrección/traducción desactivadas.")
        return cls(NullBackend())

    @property
    def is_active(self) -> bool:
        """True si hay un LLM real disponible."""
        return not isinstance(self._backend, NullBackend)

    @property
    def backend_name(self) -> str:
        return self._backend.name

    def unload(self):
        """Descarga el modelo para liberar VRAM antes de cargar Whisper."""
        if hasattr(self._backend, "unload"):
            self._backend.unload()

    # ─────────────────────────────────────────
    # Tareas de alto nivel
    # ─────────────────────────────────────────

    def correct_subtitles(self, segments: list, batch_size: int = 20) -> tuple[list, bool]:
        """
        Corrige errores de transcripción en los segmentos.
        Retorna (segmentos_corregidos, hubo_cambios).
        """
        if not self.is_active:
            return segments, False

        logger.info(f"[LLMEngine] Corrigiendo {len(segments)} segmentos con {self._backend.name}")

        SYSTEM = (
            "Eres un editor profesional de subtítulos. Tu única tarea es corregir errores "
            "evidentes de transcripción automática preservando EXACTAMENTE el formato dado. "
            "Responde SOLO con el texto corregido, sin explicaciones ni comentarios."
        )

        corrected = False
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i + batch_size]
            batch_text = "\n".join(
                f"[{s.index}|{s.start:.2f}-{s.end:.2f}] {s.text}" for s in batch
            )

            user_prompt = (
                "Corrige SOLO los errores evidentes (ortografía, puntuación, nombres propios).\n"
                "REGLAS:\n"
                "1. Mantén el formato: [NUMERO|INICIO-FIN] texto\n"
                "2. NO cambies el contenido ni el significado\n"
                "3. Si una línea está bien, cópiala igual\n"
                "4. Responde SOLO las líneas, sin texto adicional\n\n"
                f"TRANSCRIPCIÓN:\n{batch_text}"
            )

            try:
                response = self._backend.generate_chat(SYSTEM, user_prompt, max_tokens=2048, temperature=0.05)
                for line in response.strip().split("\n"):
                    m = re.match(r'\[(\d+)\|[\d.]+-[\d.]+\]\s*(.*)', line.strip())
                    if m:
                        idx, text = int(m.group(1)), m.group(2).strip()
                        for seg in batch:
                            if seg.index == idx and seg.text != text:
                                seg.text = text
                                corrected = True
                                break
                logger.info(f"[LLMEngine] Lote {i//batch_size + 1}/{-(-len(segments)//batch_size)} corregido")
            except Exception as e:
                logger.warning(f"[LLMEngine] Error en lote {i//batch_size + 1}: {e}")

        return segments, corrected

    def translate_subtitles(self, segments: list, target_language: str,
                            source_language: str = "en",
                            batch_size: int = 15) -> list:
        """
        Traduce subtítulos al idioma destino.
        Preserva timestamps y formato SRT exactos.
        """
        if not self.is_active:
            return segments

        lang_name = LANGUAGE_NAMES_ENGINE.get(target_language, target_language)
        source_name = LANGUAGE_NAMES_ENGINE.get(source_language, source_language)
        logger.info(f"[LLMEngine] Traduciendo {source_name} → {lang_name} ({len(segments)} segmentos)")

        SYSTEM = (
            f"Eres un traductor profesional especializado en subtítulos. "
            f"Traduces de {source_name} a {lang_name} con precisión y naturalidad. "
            f"Mantienes el tono, registro y puntuación del original. "
            f"Responde SOLO con los subtítulos traducidos, sin explicaciones."
        )

        for i in range(0, len(segments), batch_size):
            batch = segments[i:i + batch_size]
            batch_text = "\n".join(f"[{s.index}] {s.text}" for s in batch)

            user_prompt = (
                f"Traduce al {lang_name}.\n"
                "REGLAS ESTRICTAS:\n"
                "1. Formato: [NUMERO] texto traducido\n"
                "2. Traduce SOLO el texto, no los números\n"
                "3. Subtítulos naturales y fluidos\n"
                "4. Responde SOLO con los subtítulos\n\n"
                f"SUBTÍTULOS:\n{batch_text}"
            )

            try:
                response = self._backend.generate_chat(SYSTEM, user_prompt, max_tokens=2048, temperature=0.1)
                for line in response.strip().split("\n"):
                    m = re.match(r'\[(\d+)\]\s*(.*)', line.strip())
                    if m:
                        idx, text = int(m.group(1)), m.group(2).strip()
                        for seg in batch:
                            if seg.index == idx:
                                seg.text = text
                                break
                logger.info(f"[LLMEngine] Lote traducción {i//batch_size + 1}/{-(-len(segments)//batch_size)} OK")
            except Exception as e:
                logger.warning(f"[LLMEngine] Error traducción lote {i//batch_size + 1}: {e}")

        return segments


# Tabla de idiomas para el motor LLM
LANGUAGE_NAMES_ENGINE = {
    "es": "español", "en": "inglés", "fr": "francés",
    "de": "alemán", "it": "italiano", "pt": "portugués",
    "ja": "japonés", "zh": "chino", "ru": "ruso",
    "ar": "árabe", "ko": "coreano", "nl": "holandés",
    "pl": "polaco", "sv": "sueco", "tr": "turco",
}
