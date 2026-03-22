#!/usr/bin/env python3
"""
app_flask.py
============
App web Flask para generacion de subtitulos .srt.

Uso:
    python app_flask.py
    Abre http://localhost:5000 en el navegador.
"""

import os
import uuid
import threading
import logging
from pathlib import Path
from flask import Flask, request, jsonify, send_file, render_template_string

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2 GB max upload

UPLOAD_DIR = Path("/tmp/subtitleai_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Estado de los jobs en memoria: job_id -> dict
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


# ─────────────────────────────────────────────
# HTML de la interfaz
# ─────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SubtitleAI</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f0f13; color: #e0e0e0; min-height: 100vh; display: flex; align-items: center; justify-content: center; }
  .card { background: #1a1a24; border: 1px solid #2a2a3a; border-radius: 16px; padding: 2.5rem; width: 100%; max-width: 600px; }
  h1 { font-size: 1.6rem; font-weight: 700; color: #a78bfa; margin-bottom: 0.3rem; }
  .subtitle { color: #888; font-size: 0.9rem; margin-bottom: 2rem; }
  .form-group { margin-bottom: 1.2rem; }
  label { display: block; font-size: 0.85rem; color: #aaa; margin-bottom: 0.4rem; }
  select, input[type=text] { width: 100%; padding: 0.6rem 0.8rem; background: #111118; border: 1px solid #333; border-radius: 8px; color: #e0e0e0; font-size: 0.9rem; }
  .row { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
  .drop-zone { border: 2px dashed #333; border-radius: 12px; padding: 2rem; text-align: center; cursor: pointer; transition: border-color .2s, background .2s; margin-bottom: 1.2rem; }
  .drop-zone:hover, .drop-zone.drag { border-color: #a78bfa; background: #1e1e2e; }
  .drop-zone p { color: #888; font-size: 0.9rem; }
  .drop-zone .file-name { color: #a78bfa; font-weight: 600; margin-top: 0.4rem; }
  #fileInput { display: none; }
  .checkbox-row { display: flex; gap: 1.5rem; flex-wrap: wrap; margin-bottom: 1.2rem; }
  .checkbox-row label { display: flex; align-items: center; gap: 0.4rem; color: #ccc; font-size: 0.9rem; cursor: pointer; }
  .checkbox-row input[type=checkbox] { accent-color: #a78bfa; width: 16px; height: 16px; }
  button[type=submit] { width: 100%; padding: 0.85rem; background: linear-gradient(135deg, #7c3aed, #a78bfa); color: #fff; border: none; border-radius: 10px; font-size: 1rem; font-weight: 600; cursor: pointer; transition: opacity .2s; }
  button[type=submit]:disabled { opacity: 0.5; cursor: not-allowed; }
  #progress-box { display: none; margin-top: 1.5rem; }
  .prog-label { font-size: 0.85rem; color: #aaa; margin-bottom: 0.5rem; }
  .prog-track { background: #111118; border-radius: 99px; height: 8px; overflow: hidden; }
  .prog-bar { height: 100%; background: linear-gradient(90deg, #7c3aed, #a78bfa); border-radius: 99px; transition: width .4s; width: 0%; }
  .prog-step { font-size: 0.82rem; color: #888; margin-top: 0.5rem; min-height: 1.2em; }
  #result-box { display: none; margin-top: 1.5rem; padding: 1rem; background: #111; border: 1px solid #2a5; border-radius: 10px; }
  #result-box .ok { color: #4ade80; font-weight: 600; }
  #result-box a { display: inline-block; margin-top: 0.8rem; padding: 0.5rem 1.2rem; background: #4ade80; color: #000; border-radius: 8px; text-decoration: none; font-weight: 600; }
  #error-box { display: none; margin-top: 1.5rem; padding: 1rem; background: #1a0a0a; border: 1px solid #a33; border-radius: 10px; color: #f87171; font-size: 0.9rem; }
  .tags { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 0.8rem; }
  .tag { background: #2a2a3a; padding: 0.2rem 0.6rem; border-radius: 99px; font-size: 0.78rem; color: #aaa; }
</style>
</head>
<body>
<div class="card">
  <h1>SubtitleAI</h1>
  <p class="subtitle">Genera subtitulos .srt con IA local &mdash; Whisper + llama.cpp</p>

  <form id="form">
    <div class="drop-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
      <p>Arrastra aqui tu archivo de video o audio</p>
      <p style="font-size:.8rem;margin-top:.3rem">MP4, MKV, AVI, MOV, MP3, WAV, M4A...</p>
      <div class="file-name" id="fileName"></div>
    </div>
    <input type="file" id="fileInput" accept="video/*,audio/*,.srt">

    <div class="row">
      <div class="form-group">
        <label>Idioma del audio</label>
        <select name="language" id="language">
          <option value="auto">Auto-detectar</option>
          <option value="es" selected>Espanol</option>
          <option value="en">Ingles</option>
          <option value="fr">Frances</option>
          <option value="de">Aleman</option>
          <option value="it">Italiano</option>
          <option value="pt">Portugues</option>
          <option value="ja">Japones</option>
          <option value="zh">Chino</option>
          <option value="ru">Ruso</option>
        </select>
      </div>
      <div class="form-group">
        <label>Modelo Whisper</label>
        <select name="model" id="model">
          <option value="tiny">Tiny (rapido)</option>
          <option value="base">Base</option>
          <option value="small">Small</option>
          <option value="medium" selected>Medium (recomendado)</option>
          <option value="large-v2">Large-v2 (lento, preciso)</option>
          <option value="large-v3">Large-v3</option>
        </select>
      </div>
    </div>

    <div class="form-group">
      <label>Traducir al idioma (opcional)</label>
      <select name="translate" id="translate">
        <option value="">Sin traduccion</option>
        <option value="es">Espanol</option>
        <option value="en">Ingles</option>
        <option value="fr">Frances</option>
        <option value="de">Aleman</option>
        <option value="it">Italiano</option>
        <option value="pt">Portugues</option>
      </select>
    </div>

    <div class="checkbox-row">
      <label><input type="checkbox" name="denoise" id="denoise" checked> Reduccion de ruido</label>
      <label><input type="checkbox" name="correct" id="correct" checked> Correccion LLM</label>
      <label><input type="checkbox" name="auto_translate" id="auto_translate" checked> Auto-traducir al espanol</label>
      <label><input type="checkbox" name="diarize" id="diarize"> Identificar hablantes</label>
    </div>

    <button type="submit" id="btn">Generar subtitulos</button>
  </form>

  <div id="progress-box">
    <div class="prog-label">Procesando...</div>
    <div class="prog-track"><div class="prog-bar" id="progBar"></div></div>
    <div class="prog-step" id="progStep"></div>
  </div>

  <div id="result-box">
    <div class="ok">Subtitulos generados</div>
    <div class="tags" id="resultTags"></div>
    <a id="downloadBtn" href="#">Descargar .srt</a>
  </div>

  <div id="error-box" id="errorBox"></div>
</div>

<script>
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
let selectedFile = null;

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) {
    selectedFile = fileInput.files[0];
    document.getElementById('fileName').textContent = selectedFile.name;
  }
});

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag');
  if (e.dataTransfer.files[0]) {
    selectedFile = e.dataTransfer.files[0];
    document.getElementById('fileName').textContent = selectedFile.name;
    fileInput.files = e.dataTransfer.files;
  }
});

document.getElementById('form').addEventListener('submit', async e => {
  e.preventDefault();
  if (!selectedFile) { alert('Selecciona un archivo primero'); return; }

  const btn = document.getElementById('btn');
  btn.disabled = true;
  document.getElementById('progress-box').style.display = 'block';
  document.getElementById('result-box').style.display = 'none';
  document.getElementById('error-box').style.display = 'none';

  const fd = new FormData();
  fd.append('file', selectedFile);
  fd.append('language', document.getElementById('language').value);
  fd.append('model', document.getElementById('model').value);
  fd.append('translate', document.getElementById('translate').value);
  fd.append('denoise', document.getElementById('denoise').checked ? '1' : '0');
  fd.append('correct', document.getElementById('correct').checked ? '1' : '0');
  fd.append('auto_translate', document.getElementById('auto_translate').checked ? '1' : '0');
  fd.append('diarize', document.getElementById('diarize').checked ? '1' : '0');

  const res = await fetch('/submit', { method: 'POST', body: fd });
  const data = await res.json();
  if (!data.job_id) { showError('Error al iniciar el job'); btn.disabled = false; return; }

  pollJob(data.job_id, btn);
});

function pollJob(jobId, btn) {
  const interval = setInterval(async () => {
    const res = await fetch('/status/' + jobId);
    const data = await res.json();

    document.getElementById('progBar').style.width = data.pct + '%';
    document.getElementById('progStep').textContent = data.step || '';

    if (data.status === 'done') {
      clearInterval(interval);
      btn.disabled = false;
      showResult(data, jobId);
    } else if (data.status === 'error') {
      clearInterval(interval);
      btn.disabled = false;
      showError(data.error || 'Error desconocido');
    }
  }, 800);
}

function showResult(data, jobId) {
  document.getElementById('result-box').style.display = 'block';
  const tags = document.getElementById('resultTags');
  tags.innerHTML = '';
  if (data.segments) addTag(tags, data.segments + ' segmentos');
  if (data.language) addTag(tags, 'Idioma: ' + data.language);
  if (data.quality) addTag(tags, 'Calidad: ' + data.quality);
  if (data.corrected) addTag(tags, 'LLM corregido');
  if (data.translated) addTag(tags, 'Traducido: ' + data.target_language);
  document.getElementById('downloadBtn').href = '/download/' + jobId;
}

function addTag(container, text) {
  const span = document.createElement('span');
  span.className = 'tag';
  span.textContent = text;
  container.appendChild(span);
}

function showError(msg) {
  const box = document.getElementById('error-box');
  box.style.display = 'block';
  box.textContent = 'Error: ' + msg;
}
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────
# Rutas Flask
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/submit", methods=["POST"])
def submit():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    job_id = str(uuid.uuid4())[:8]
    suffix = Path(f.filename).suffix.lower() or ".mp4"
    input_path = UPLOAD_DIR / f"{job_id}{suffix}"
    output_path = UPLOAD_DIR / f"{job_id}.srt"

    f.save(str(input_path))

    options = {
        "language":       request.form.get("language", "auto"),
        "model":          request.form.get("model", "medium"),
        "translate":      request.form.get("translate", "") or None,
        "denoise":        request.form.get("denoise", "1") == "1",
        "correct":        request.form.get("correct", "1") == "1",
        "auto_translate": request.form.get("auto_translate", "1") == "1",
        "diarize":        request.form.get("diarize", "0") == "1",
    }

    with jobs_lock:
        jobs[job_id] = {
            "status": "pending",
            "pct": 0,
            "step": "En cola...",
            "input_path": str(input_path),
            "output_path": str(output_path),
        }

    thread = threading.Thread(
        target=_run_job,
        args=(job_id, str(input_path), str(output_path), options),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"status": "not_found"}), 404
    return jsonify(job)


@app.route("/download/<job_id>")
def download(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job or job.get("status") != "done":
        return jsonify({"error": "Not ready"}), 404

    srt_path = job["output_path"]
    if not Path(srt_path).exists():
        return jsonify({"error": "File not found"}), 404

    original_name = Path(job["input_path"]).stem + ".srt"
    return send_file(
        srt_path,
        as_attachment=True,
        download_name=original_name,
        mimetype="text/plain",
    )


# ─────────────────────────────────────────────
# Worker del pipeline
# ─────────────────────────────────────────────

def _update_job(job_id: str, **kwargs):
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id].update(kwargs)


def _run_job(job_id: str, input_path: str, output_path: str, options: dict):
    _update_job(job_id, status="running", pct=1, step="Iniciando pipeline...")

    try:
        from core.pipeline import run_pipeline, segments_to_srt

        def progress(step, pct):
            _update_job(job_id, step=step, pct=pct)

        result = run_pipeline(
            audio_path=input_path,
            language=options["language"],
            model_size=options["model"],
            enable_denoise=options["denoise"],
            enable_diarization=options["diarize"],
            enable_correction=options["correct"],
            enable_translation=bool(options["translate"]),
            target_language=options["translate"],
            auto_translate_to_spanish=options["auto_translate"],
            progress_callback=progress,
        )

        srt_content = segments_to_srt(result.segments)
        Path(output_path).write_text(srt_content, encoding="utf-8")

        quality_label = result.audio_quality.quality_label if result.audio_quality else None

        _update_job(
            job_id,
            status="done",
            pct=100,
            step="Completado",
            segments=len(result.segments),
            language=f"{result.language} ({result.language_probability:.0%})",
            quality=quality_label,
            corrected=result.corrected,
            translated=result.translated,
            target_language=result.target_language,
        )

    except Exception as e:
        logging.exception(f"[Job {job_id}] Error en pipeline")
        _update_job(job_id, status="error", error=str(e), pct=0)
    finally:
        # Limpiar archivo de entrada
        try:
            Path(input_path).unlink(missing_ok=True)
        except Exception:
            pass


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("""
  SubtitleAI — App Web
  Abre http://localhost:5000 en tu navegador
    """)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
