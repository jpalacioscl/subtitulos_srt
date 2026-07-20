#!/usr/bin/env bash
# run_web.sh — Lanzador de la app web Flask
# Abre http://localhost:5000 en el navegador

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$DIR/venv/bin/python3"

if [ ! -x "$PYTHON" ]; then
    echo "Error: venv no encontrado. Ejecuta primero: python3 setup_blackwell.py" >&2
    exit 1
fi

exec "$PYTHON" "$DIR/app_flask.py"
