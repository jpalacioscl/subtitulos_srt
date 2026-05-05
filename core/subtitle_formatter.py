"""
core/subtitle_formatter.py
==========================
Post-procesador técnico de subtítulos según la Guía Maestra de Subtitulado.

Reglas de la guía implementadas:
  - CPL  : ≤42 caracteres por línea, máximo 2 líneas por bloque
  - CPS  : ≤17 caracteres/segundo (velocidad de lectura para adultos)
  - Dur  : entre 1 s y 7 s por subtítulo
  - Gap  : ≥100 ms de silencio entre subtítulos consecutivos
  - Líneas: segmentación en frontera sintáctica (no romper artículo+sustantivo,
            preposición+objeto, sustantivo+adjetivo, verbo+clítico/negación)
  - Pirámide: línea inferior más larga (o igual) que la superior
  - Puntuación: sin punto final tras ? o !; guion + espacio en diálogos
"""

import re
import logging
from dataclasses import replace as dc_replace

logger = logging.getLogger(__name__)

# ── Constantes de la Guía Maestra ─────────────────────────────────────────────
MAX_CPL = 42       # caracteres por línea (estándar Netflix/Warner)
MAX_CPS = 17       # caracteres por segundo (adultos)
MIN_DUR = 1.0      # duración mínima en segundos
MAX_DUR = 7.0      # duración máxima en segundos
MIN_GAP = 0.100    # separación mínima entre subtítulos (100 ms)


# ── Léxico para reglas de segmentación (español) ──────────────────────────────
_ARTICLES = frozenset({
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    "al", "del",
})
_PREPOSITIONS = frozenset({
    "a", "ante", "bajo", "cabe", "con", "contra", "de", "desde",
    "durante", "en", "entre", "hacia", "hasta", "mediante", "para",
    "por", "según", "sin", "so", "sobre", "tras", "versus", "vía",
})
_NEGATIONS = frozenset({"no", "ni", "nunca", "jamás", "tampoco"})
_CLITICS = frozenset({
    "me", "te", "se", "le", "les", "lo", "la", "los", "las",
    "nos", "os",
})
_CONJUNCTIONS = frozenset({
    "y", "e", "o", "u", "pero", "mas", "sino", "aunque", "porque",
    "que", "cuando", "como", "donde", "si", "ni", "ya", "pues",
    "mientras", "entonces", "luego", "además", "también",
})


def _visible_len(text: str) -> int:
    """Longitud visible, ignorando tags de formato HTML (<i>, <b>, etc.)."""
    return len(re.sub(r'<[^>]+>', '', text))


def _find_split_point(words: list, max_chars: int) -> int:
    """
    Devuelve el índice (exclusivo) del mejor corte de la lista de palabras
    tal que la primera línea no supere max_chars.

    Prioriza:
      +50  coma/punto y coma/dos puntos al final de la palabra anterior
      +30  conjunción al inicio de la parte siguiente
      +10  cercanía al centro (estructura equilibrada)
      -100 artículo justo antes del corte
      -90  preposición justo antes del corte
      -80  negación/clítico justo después del corte
    """
    candidates = []
    acc = 0
    for i, w in enumerate(words):
        acc += len(w) + (1 if i > 0 else 0)
        if acc > max_chars:
            break
        if i < len(words) - 1:
            candidates.append(i + 1)

    if not candidates:
        return max(1, len(words) // 2)

    mid = len(words) / 2

    def _score(idx: int) -> int:
        prev = words[idx - 1].lower().rstrip(",:;.!?—…\"')")
        nxt  = words[idx].lower().lstrip("¿¡\"'(")
        if prev in _ARTICLES:
            return -100
        if prev in _PREPOSITIONS:
            return -90
        if nxt in _NEGATIONS | _CLITICS:
            return -80
        pts = 0
        if words[idx - 1].endswith((",", ";", ":", "—")):
            pts += 50
        if nxt in _CONJUNCTIONS:
            pts += 30
        pts += max(0, 10 - int(abs(idx - mid) * 2))
        return pts

    return max(candidates, key=_score)


def split_into_lines(text: str, max_cpl: int = MAX_CPL) -> str:
    """
    Formatea el texto en 1 ó 2 líneas de ≤max_cpl caracteres respetando
    las fronteras sintácticas y la estructura en pirámide.
    """
    text = text.strip()
    if _visible_len(text) <= max_cpl:
        return text

    words = text.split()
    total = sum(len(w) for w in words) + max(0, len(words) - 1)

    split_at = _find_split_point(words, max_cpl)
    line1 = " ".join(words[:split_at])
    line2 = " ".join(words[split_at:])

    # Corregir si alguna línea sigue excediendo el límite
    if _visible_len(line1) > max_cpl or _visible_len(line2) > max_cpl:
        mid = len(words) // 2
        line1 = " ".join(words[:mid])
        line2 = " ".join(words[mid:])

    # Estructura en pirámide: mover última palabra de línea 1 a línea 2
    # si con eso línea 2 sigue siendo ≤ max_cpl y línea 1 queda más corta
    w1 = line1.split()
    if len(w1) > 1:
        cand1 = " ".join(w1[:-1])
        cand2 = w1[-1] + " " + line2
        if (_visible_len(cand2) <= max_cpl
                and _visible_len(cand2) >= _visible_len(cand1)):
            line1, line2 = cand1, cand2

    return f"{line1}\n{line2}"


def _min_dur_for_text(text: str, max_cps: int = MAX_CPS) -> float:
    chars = _visible_len(text.replace("\n", " "))
    return max(MIN_DUR, chars / max_cps)


def apply_punctuation_rules(text: str) -> str:
    """
    Reglas ortotipográficas de la guía:
    - Sin punto final tras ? o !
    - Guion + espacio en líneas de diálogo (- texto)
    """
    # No punto final tras cierre de interrogación/exclamación
    text = re.sub(r'([?!])\.$', r'\1', text)
    text = re.sub(r'([?!])\.\s+', r'\1 ', text)
    # Guion de diálogo: asegurar espacio después del guion
    text = re.sub(r'^-(?!\s)', '- ', text, flags=re.MULTILINE)
    return text.strip()


def _split_long_segment(seg, max_cpl: int, max_cps: int) -> list:
    """
    Divide un segmento cuyo texto supera 2×max_cpl en múltiples segmentos
    con duración proporcional al número de caracteres.
    """
    text = seg.text.strip()
    if _visible_len(text) <= max_cpl * 2:
        return [seg]

    # Intentar dividir por puntuación fuerte
    parts = re.split(r'(?<=[.!?…])\s+', text)
    if len(parts) < 2:
        parts = re.split(r',\s+', text, maxsplit=1)
    if len(parts) < 2:
        words = text.split()
        mid = len(words) // 2
        parts = [" ".join(words[:mid]), " ".join(words[mid:])]

    parts = [p.strip() for p in parts if p.strip()]
    total_chars = sum(_visible_len(p) for p in parts) or 1
    total_dur   = seg.end - seg.start
    result      = []
    t_start     = seg.start

    for i, part in enumerate(parts):
        ratio   = _visible_len(part) / total_chars
        dur     = max(_min_dur_for_text(part, max_cps), total_dur * ratio)
        t_end   = min(seg.end, t_start + dur)
        result.append(dc_replace(seg, index=seg.index + i,
                                 start=t_start, end=t_end, text=part))
        t_start = t_end

    return result


def format_segments(segments: list,
                    max_cpl: int = MAX_CPL,
                    max_cps: int = MAX_CPS) -> list:
    """
    Aplica el conjunto completo de reglas técnicas de la Guía Maestra.

    Pasos:
      1. Reglas de puntuación
      2. Dividir segmentos demasiado largos para 2 líneas
      3. Formatear texto en 1-2 líneas (CPL)
      4. Extender duración mínima por CPS
      5. Aplicar duración mínima/máxima absoluta
      6. Asegurar gap ≥ 100 ms entre subtítulos
      7. Renumerar índices desde 1
    """
    # ── 1. Puntuación ─────────────────────────────────────────────────────────
    segs = [dc_replace(s, text=apply_punctuation_rules(s.text)) for s in segments]

    # ── 2. Dividir segmentos muy largos ───────────────────────────────────────
    expanded = []
    for s in segs:
        expanded.extend(_split_long_segment(s, max_cpl=max_cpl, max_cps=max_cps))

    # ── 3. Formatear en 1-2 líneas ────────────────────────────────────────────
    lined = [dc_replace(s, text=split_into_lines(s.text, max_cpl=max_cpl))
             for s in expanded]

    # ── 4-6. Timing en una sola pasada (CPS + dur + gap) ─────────────────────
    # Extender duración solo hasta donde permite el inicio del segmento siguiente,
    # luego recortar si aun así solapan.  Cambiar start_time nunca (sync de audio).
    result = list(lined)
    for i, s in enumerate(result):
        min_needed = max(MIN_DUR, _min_dur_for_text(s.text, max_cps))
        max_allowed = s.start + MAX_DUR

        # Techo: lo antes que empieza el siguiente subtítulo menos el gap
        if i + 1 < len(result):
            max_allowed = min(max_allowed, result[i + 1].start - MIN_GAP)

        new_end = s.end
        if new_end - s.start < min_needed:
            new_end = s.start + min_needed   # extender para CPS / min_dur
        if new_end > max_allowed:
            new_end = max_allowed            # respetar siguiente segmento
        if new_end < s.start + 0.1:
            new_end = s.start + 0.1         # al menos 100 ms de duración

        result[i] = dc_replace(s, end=new_end)

    # ── 7. Renumerar ──────────────────────────────────────────────────────────
    final = [dc_replace(s, index=i + 1) for i, s in enumerate(result)]

    n_in, n_out = len(segments), len(final)
    logger.info(
        f"[Formatter] {n_in} seg → {n_out} seg | "
        f"CPL≤{max_cpl} | CPS≤{max_cps} | gap≥{MIN_GAP*1000:.0f}ms | "
        f"dur {MIN_DUR}-{MAX_DUR}s"
    )
    return final
