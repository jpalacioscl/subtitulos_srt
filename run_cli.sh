#!/usr/bin/env bash
# run_cli.sh — Lanzador del CLI de subtitulos
# Uso: ./run_cli.sh video.mp4 [opciones]
#      ./run_cli.sh --help

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$DIR/venv/bin/python3" "$DIR/subtitles_cli.py" "$@"
