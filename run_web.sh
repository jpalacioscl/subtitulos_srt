#!/usr/bin/env bash
# run_web.sh — Lanzador de la app web Flask
# Abre http://localhost:5000 en el navegador

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/venv/bin/activate"
python "$DIR/app_flask.py"
