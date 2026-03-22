"""
core/model_manager.py
=====================
Gestor de modelos GGUF para llama.cpp.

Responsabilidades:
  - Listar modelos .gguf disponibles en el directorio local
  - Descargar modelos desde HuggingFace Hub
  - Mostrar información de cada modelo (tamaño, cuantización, estado)
  - Recomendar el modelo óptimo según el hardware detectado

Modelos recomendados para Alienware RTX 5060 (8GB VRAM + 64GB RAM):
  ┌──────────────────────────────────────┬────────┬──────────┬────────────────┐
  │ Modelo                               │ Tamaño │ VRAM     │ Calidad        │
  ├──────────────────────────────────────┼────────┼──────────┼────────────────┤
  │ Llama-3.1-8B-Instruct-Q4_K_M        │ 4.9 GB │ ~5.5 GB  │ ⭐⭐⭐⭐ Rec. │
  │ Llama-3.1-8B-Instruct-Q5_K_M        │ 5.7 GB │ ~6.2 GB  │ ⭐⭐⭐⭐⭐     │
  │ Mistral-7B-Instruct-v0.3-Q4_K_M     │ 4.4 GB │ ~5.0 GB  │ ⭐⭐⭐⭐       │
  │ Phi-3-mini-4k-instruct-Q4_K_M       │ 2.2 GB │ ~2.8 GB  │ ⭐⭐⭐         │
  │ Llama-3.3-70B-Instruct-Q2_K         │ 26 GB  │ RAM only │ ⭐⭐⭐⭐⭐+    │
  └──────────────────────────────────────┴────────┴──────────┴────────────────┘
"""

import os
import json
import logging
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MODELS_DIR = Path.home() / ".subtitle_ai" / "models"
DEFAULT_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# Catálogo de modelos recomendados
# ─────────────────────────────────────────────

RECOMMENDED_MODELS = [
    {
        "id": "llama3.1-8b-q4",
        "name": "Llama 3.1 8B Instruct Q4_K_M",
        "filename": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "repo": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "size_gb": 4.92,
        "vram_gb": 5.5,
        "quality": 4,
        "speed": 4,
        "description": "Recomendado para RTX 5060. Excelente balance calidad/velocidad.",
        "tag": "⭐ Recomendado",
        "min_vram_gb": 5.5,
        "min_ram_gb": 8,
    },
    {
        "id": "llama3.1-8b-q5",
        "name": "Llama 3.1 8B Instruct Q5_K_M",
        "filename": "Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf",
        "repo": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "size_gb": 5.73,
        "vram_gb": 6.2,
        "quality": 5,
        "speed": 3,
        "description": "Mayor precisión que Q4. Cabe en 8GB VRAM con margen ajustado.",
        "tag": "Alta calidad",
        "min_vram_gb": 6.2,
        "min_ram_gb": 8,
    },
    {
        "id": "mistral-7b-q4",
        "name": "Mistral 7B Instruct v0.3 Q4_K_M",
        "filename": "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
        "repo": "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
        "size_gb": 4.37,
        "vram_gb": 5.0,
        "quality": 4,
        "speed": 5,
        "description": "Muy rápido, excelente para traducción. Menos VRAM que Llama3.",
        "tag": "Más rápido",
        "min_vram_gb": 5.0,
        "min_ram_gb": 8,
    },
    {
        "id": "phi3-mini-q4",
        "name": "Phi-3 Mini 4K Instruct Q4_K_M",
        "filename": "Phi-3-mini-4k-instruct-Q4_K_M.gguf",
        "repo": "microsoft/Phi-3-mini-4k-instruct-gguf",
        "size_gb": 2.18,
        "vram_gb": 2.8,
        "quality": 3,
        "speed": 5,
        "description": "Ultra ligero. Ideal para liberar VRAM para Whisper large-v2 simultáneo.",
        "tag": "Ligero",
        "min_vram_gb": 2.8,
        "min_ram_gb": 4,
    },
    {
        "id": "llama3.3-70b-q2",
        "name": "Llama 3.3 70B Instruct Q2_K",
        "filename": "Llama-3.3-70B-Instruct-Q2_K.gguf",
        "repo": "bartowski/Llama-3.3-70B-Instruct-GGUF",
        "size_gb": 26.0,
        "vram_gb": 0,
        "quality": 5,
        "speed": 1,
        "description": "Máxima calidad, corre en RAM (64GB necesarios). Sin GPU. Lento pero preciso.",
        "tag": "Máxima calidad (CPU)",
        "min_vram_gb": 0,
        "min_ram_gb": 32,
    },
]


# ─────────────────────────────────────────────
# Estructuras de datos
# ─────────────────────────────────────────────

@dataclass
class ModelInfo:
    id: str
    name: str
    filename: str
    path: str
    size_gb: float
    is_downloaded: bool
    vram_gb: float
    quality: int        # 1-5
    speed: int          # 1-5
    description: str
    tag: str
    compatible: bool    # según hardware detectado
    recommended: bool


# ─────────────────────────────────────────────
# ModelManager
# ─────────────────────────────────────────────

class ModelManager:
    """
    Gestiona el ciclo de vida de los modelos GGUF locales.
    """

    def __init__(self, models_dir: Optional[str] = None):
        self.models_dir = Path(models_dir) if models_dir else DEFAULT_MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def list_models(self) -> list[ModelInfo]:
        """
        Lista todos los modelos del catálogo, indicando cuáles están
        descargados y cuáles son compatibles con el hardware actual.
        """
        hw = self._detect_hardware()
        downloaded_files = {f.name: f for f in self.models_dir.glob("*.gguf")}

        # Incluir también GGUFs locales no en el catálogo
        catalogued_filenames = {m["filename"] for m in RECOMMENDED_MODELS}
        extra_files = [f for f in downloaded_files.values()
                       if f.name not in catalogued_filenames]

        models = []

        for m in RECOMMENDED_MODELS:
            path = self.models_dir / m["filename"]
            is_dl = m["filename"] in downloaded_files
            compatible = (
                (m["min_vram_gb"] == 0 or hw["vram_gb"] >= m["min_vram_gb"])
                and hw["ram_gb"] >= m["min_ram_gb"]
            )
            # El recomendado es el de mayor quality que sea compatible con el hardware
            models.append(ModelInfo(
                id=m["id"],
                name=m["name"],
                filename=m["filename"],
                path=str(path),
                size_gb=m["size_gb"],
                is_downloaded=is_dl,
                vram_gb=m["vram_gb"],
                quality=m["quality"],
                speed=m["speed"],
                description=m["description"],
                tag=m["tag"],
                compatible=compatible,
                recommended=False,
            ))

        # Marcar el recomendado: compatible + mayor quality + descargado preferido
        compatible_downloaded = [m for m in models if m.compatible and m.is_downloaded]
        compatible_all = [m for m in models if m.compatible]

        best = None
        if compatible_downloaded:
            best = max(compatible_downloaded, key=lambda m: m.quality)
        elif compatible_all:
            best = max(compatible_all, key=lambda m: m.quality)

        if best:
            best.recommended = True

        # Añadir GGUFs locales extra
        for f in extra_files:
            size_gb = f.stat().st_size / 1024**3
            models.append(ModelInfo(
                id=f"local_{f.stem}",
                name=f.stem,
                filename=f.name,
                path=str(f),
                size_gb=round(size_gb, 2),
                is_downloaded=True,
                vram_gb=0,
                quality=3,
                speed=3,
                description="Modelo local personalizado",
                tag="Local",
                compatible=True,
                recommended=False,
            ))

        return models

    def get_best_available(self) -> Optional[str]:
        """
        Retorna la ruta al mejor modelo descargado y compatible.
        None si ninguno está disponible.
        """
        models = self.list_models()
        available = [m for m in models if m.is_downloaded and m.compatible]
        if not available:
            return None
        best = max(available, key=lambda m: m.quality)
        return best.path

    def download_model(self, model_id: str, progress_callback=None) -> str:
        """
        Descarga un modelo del catálogo desde HuggingFace Hub.
        Muestra progreso en tiempo real.

        Requiere: pip install huggingface-hub
        """
        model_spec = next((m for m in RECOMMENDED_MODELS if m["id"] == model_id), None)
        if not model_spec:
            raise ValueError(f"Modelo '{model_id}' no encontrado en el catálogo")

        dest_path = self.models_dir / model_spec["filename"]
        if dest_path.exists():
            logger.info(f"[ModelManager] Modelo ya descargado: {dest_path}")
            return str(dest_path)

        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise RuntimeError("huggingface-hub no instalado. Instala con: pip install huggingface-hub")

        logger.info(f"[ModelManager] Descargando {model_spec['name']} ({model_spec['size_gb']:.1f} GB)...")
        logger.info(f"[ModelManager] Repo: {model_spec['repo']}")

        if progress_callback:
            progress_callback(f"Descargando {model_spec['name']}...", 0)

        try:
            # hf_hub_download descarga con caché y progreso
            cached_path = hf_hub_download(
                repo_id=model_spec["repo"],
                filename=model_spec["filename"],
                local_dir=str(self.models_dir),
                local_dir_use_symlinks=False,
            )
            logger.info(f"[ModelManager] ✓ Descargado: {cached_path}")
            if progress_callback:
                progress_callback(f"✓ {model_spec['name']} descargado", 100)
            return cached_path

        except Exception as e:
            raise RuntimeError(f"Error descargando modelo: {e}\n"
                               f"Descarga manual desde:\n"
                               f"https://huggingface.co/{model_spec['repo']}/resolve/main/{model_spec['filename']}")

    def delete_model(self, model_id: str) -> bool:
        """Elimina un modelo descargado para liberar espacio en disco."""
        model_spec = next((m for m in RECOMMENDED_MODELS if m["id"] == model_id), None)
        if not model_spec:
            return False
        path = self.models_dir / model_spec["filename"]
        if path.exists():
            path.unlink()
            logger.info(f"[ModelManager] Modelo eliminado: {path.name}")
            return True
        return False

    def disk_usage(self) -> dict:
        """Retorna uso de disco del directorio de modelos."""
        total = sum(f.stat().st_size for f in self.models_dir.glob("*.gguf"))
        return {
            "total_gb": round(total / 1024**3, 2),
            "models_dir": str(self.models_dir),
            "count": len(list(self.models_dir.glob("*.gguf"))),
        }

    def _detect_hardware(self) -> dict:
        """Detecta VRAM y RAM disponibles."""
        import psutil
        ram_gb = psutil.virtual_memory().total / 1024**3

        vram_gb = 0
        try:
            import torch
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        except Exception:
            pass

        return {"vram_gb": vram_gb, "ram_gb": ram_gb}
