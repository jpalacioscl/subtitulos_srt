"""
Configuración centralizada de logging para SubtitleAI.

Uso:
    from core.logging_setup import setup_logging
    setup_logging()           # INFO en consola, DEBUG en archivo
    setup_logging(verbose=True)  # DEBUG en consola también
"""

import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path


_CONSOLE_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
_FILE_FORMAT    = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
_DATE_FORMAT    = "%H:%M:%S"


def setup_logging(verbose: bool = False, log_dir: str | Path = "logs") -> Path:
    """
    Configura logging con dos destinos:
      - Consola: INFO (o DEBUG si verbose=True)
      - Archivo: DEBUG siempre, rotación por ejecución

    Retorna la ruta al archivo de log de esta sesión.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)

    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"subtitulos_{session_ts}.log"

    root = logging.getLogger()
    # Evitar duplicar handlers si se llama dos veces
    if root.handlers:
        return log_file

    root.setLevel(logging.DEBUG)

    # ── Consola ──────────────────────────────────────────────
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter(_CONSOLE_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(console)

    # ── Archivo ──────────────────────────────────────────────
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(_FILE_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
    root.addHandler(file_handler)

    # Silenciar loggers ruidosos de terceros
    for noisy in ("httpx", "httpcore", "urllib3", "filelock", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.info(f"[Log] Sesión iniciada → {log_file}")
    return log_file
