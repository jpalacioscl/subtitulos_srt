#!/usr/bin/env bash
# run_cli.sh — Lanzador del CLI de subtitulos
# Uso: ./run_cli.sh video.mp4 [opciones]
#      ./run_cli.sh --help

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$DIR/venv/bin/python3"

if [ ! -x "$PYTHON" ]; then
    echo "Error: venv no encontrado. Ejecuta primero: python3 setup_blackwell.py" >&2
    exit 1
fi

exec "$PYTHON" "$DIR/subtitles_cli.py" "$@"
