#!/usr/bin/env bash
# run_web.sh — Lanzador de la app web Flask
# Abre http://localhost:5000 en el navegador

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$DIR/venv/bin/python3" "$DIR/app_flask.py"
