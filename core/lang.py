"""
lang.py
=======
Utilidades compartidas para el etiquetado de idioma en nombres de archivo .srt.

Usadas tanto por el CLI (subtitles_cli.py) como por la app web (app_flask.py)
para construir el sufijo de idioma del nombre de salida (p. ej. "_sp", "_en").
"""

_LANG_DISPLAY = {
    "es": "sp", "en": "en", "fr": "fr", "de": "de",
    "it": "it", "pt": "pt", "ja": "ja", "zh": "zh", "ru": "ru",
}


def lang_tag(code: str) -> str:
    return _LANG_DISPLAY.get(code.lower(), code.lower())


def build_lang_suffix(source: str, target: str | None) -> str:
    lang = lang_tag(target) if target else lang_tag(source)
    return f"_{lang}"


def build_srt_filename(base: str, source: str, target: str | None, max_len: int = 120) -> str:
    """Nombre .srt canonico: base saneada + sufijo de idioma + extension."""
    safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in base)[:max_len].strip().strip("_")
    if not safe:
        safe = "subtitulos"
    suffix = build_lang_suffix(source, target) if source else ""
    return f"{safe}{suffix}.srt"
