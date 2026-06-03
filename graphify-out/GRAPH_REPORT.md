# Graph Report - .  (2026-06-03)

## Corpus Check
- 14 files · ~13,977 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 231 nodes · 425 edges · 18 communities (8 shown, 10 thin omitted)
- Extraction: 98% EXTRACTED · 0% INFERRED · 2% AMBIGUOUS · INFERRED: 2 edges (avg confidence: 0.9)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_LLM Engine & Backends|LLM Engine & Backends]]
- [[_COMMUNITY_Audio Processing Pipeline|Audio Processing Pipeline]]
- [[_COMMUNITY_Subtitling Standards & Concepts|Subtitling Standards & Concepts]]
- [[_COMMUNITY_Flask Web Interface|Flask Web Interface]]
- [[_COMMUNITY_Model Manager|Model Manager]]
- [[_COMMUNITY_Subtitle Formatter|Subtitle Formatter]]
- [[_COMMUNITY_Setup & Installation|Setup & Installation]]
- [[_COMMUNITY_LLM Backend Selection|LLM Backend Selection]]
- [[_COMMUNITY_Claude Permissions Config|Claude Permissions Config]]
- [[_COMMUNITY_CLI Launcher|CLI Launcher]]
- [[_COMMUNITY_Web Launcher|Web Launcher]]
- [[_COMMUNITY_Flask App Instance|Flask App Instance]]
- [[_COMMUNITY_LLM Engine Instance|LLM Engine Instance]]
- [[_COMMUNITY_Model Manager Instance|Model Manager Instance]]
- [[_COMMUNITY_Denoise Stage|Denoise Stage]]
- [[_COMMUNITY_Transcription Stage|Transcription Stage]]
- [[_COMMUNITY_CLI Shell Script|CLI Shell Script]]
- [[_COMMUNITY_Web Shell Script|Web Shell Script]]

## God Nodes (most connected - your core abstractions)
1. `run_pipeline()` - 20 edges
2. `str` - 16 edges
3. `run_pipeline() Main Pipeline Orchestrator` - 16 edges
4. `main()` - 13 edges
5. `LlamaCppBackend` - 12 edges
6. `_run_job()` - 10 edges
7. `LLMBackend` - 10 edges
8. `OllamaBackend` - 10 edges
9. `segments_to_srt()` - 10 edges
10. `str` - 9 edges

## Surprising Connections (you probably didn't know these)
- `_build_lang_suffix()` --semantically_similar_to--> `main()`  [INFERRED] [semantically similar]
  app_flask.py → subtitles_cli.py
- `SubtitleAI README Documentation` --references--> `run_pipeline() Main Pipeline Orchestrator`  [EXTRACTED]
  README.md → core/pipeline.py
- `format_segments() Master Guide Post-processor` --implements--> `CPL Rule: Max 42 Characters Per Line`  [EXTRACTED]
  core/subtitle_formatter.py → guia_maestra_subtitulos.md
- `format_segments() Master Guide Post-processor` --implements--> `CPS Rule: Max 17-20 Characters Per Second`  [EXTRACTED]
  core/subtitle_formatter.py → guia_maestra_subtitulos.md
- `format_segments() Master Guide Post-processor` --implements--> `Duration Rule: 1s Min, 6-7s Max Per Subtitle`  [EXTRACTED]
  core/subtitle_formatter.py → guia_maestra_subtitulos.md

## Import Cycles
- None detected.

## Communities (18 total, 10 thin omitted)

### Community 0 - "LLM Engine & Backends"
Cohesion: 0.09
Nodes (43): core/llm_engine.py ================== Motor LLM unificado con soporte para llama, Motor LLM unificado. Selecciona automáticamente el mejor backend     disponible, analyze_audio_quality(), assign_speakers(), AudioQualityReport, correct_with_llm(), denoise_audio(), _denoise_with_strength() (+35 more)

### Community 1 - "Audio Processing Pipeline"
Cohesion: 0.08
Nodes (28): ABC, detect_llama_cpp_config(), LlamaCppBackend, LLMBackend, NullBackend, OllamaBackend, bool, float (+20 more)

### Community 2 - "Subtitling Standards & Concepts"
Cohesion: 0.08
Nodes (33): Adaptive Denoise Strength Based on SNR Quality, CUDA OOM Fallback to CPU int8, VRAM Unload Strategy: LLM before Whisper, CPL Rule: Max 42 Characters Per Line, CPS Rule: Max 17-20 Characters Per Second, Guia Maestra Subtitulos Professional Standards Guide, Duration Rule: 1s Min, 6-7s Max Per Subtitle, Gap Rule: Min 100ms Between Subtitles (+25 more)

### Community 3 - "Flask Web Interface"
Cohesion: 0.11
Nodes (24): _build_lang_suffix(), download(), _lang_tag(), str, _run_job(), submit(), _update_job(), youtube_qualities() (+16 more)

### Community 4 - "Model Manager"
Cohesion: 0.14
Nodes (11): ModelInfo, bool, str, core/model_manager.py ===================== Gestor de modelos GGUF para llama.cp, Gestiona el ciclo de vida de los modelos GGUF locales., Lista todos los modelos del catálogo, indicando cuáles están         descargados, Retorna la ruta al mejor modelo descargado y compatible.         None si ninguno, Descarga un modelo del catálogo desde HuggingFace Hub.         Muestra progreso (+3 more)

### Community 5 - "Subtitle Formatter"
Cohesion: 0.22
Nodes (17): apply_punctuation_rules(), _find_split_point(), format_segments(), _min_dur_for_text(), float, int, str, core/subtitle_formatter.py ========================== Post-procesador técnico de (+9 more)

### Community 6 - "Setup & Installation"
Cohesion: 0.45
Nodes (16): check_python(), download_recommended_model(), err(), info(), install_faster_whisper(), install_llama_cpp(), install_pipeline_deps(), install_pytorch() (+8 more)

### Community 7 - "LLM Backend Selection"
Cohesion: 0.18
Nodes (13): LLM Backend Priority: llama.cpp > Ollama > Null, LLMEngine.auto_detect() Backend Auto-selector, detect_llama_cpp_config() Hardware Config Detector, LlamaCppBackend llama-cpp-python Backend, LLMBackend Abstract Base Class, NullBackend No-op Fallback, OllamaBackend Ollama REST Backend, ModelManager._detect_hardware() Hardware Detector (+5 more)

## Ambiguous Edges - Review These
- `llm_engine.py` → `AudioQualityReport`  [AMBIGUOUS]
  core/pipeline.py · relation: uses
- `llm_engine.py` → `PipelineResult`  [AMBIGUOUS]
  core/pipeline.py · relation: uses
- `llm_engine.py` → `bool`  [AMBIGUOUS]
  core/pipeline.py · relation: uses
- `llm_engine.py` → `float`  [AMBIGUOUS]
  core/pipeline.py · relation: uses
- `llm_engine.py` → `int`  [AMBIGUOUS]
  core/pipeline.py · relation: uses
- `llm_engine.py` → `str`  [AMBIGUOUS]
  core/pipeline.py · relation: uses
- `llm_engine.py` → `Segment`  [AMBIGUOUS]
  core/pipeline.py · relation: uses

## Knowledge Gaps
- **31 isolated node(s):** `allow`, `bool`, `float`, `run_cli.sh script`, `run_web.sh script` (+26 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **10 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `llm_engine.py` and `AudioQualityReport`?**
  _Edge tagged AMBIGUOUS (relation: uses) - confidence is low._
- **What is the exact relationship between `llm_engine.py` and `PipelineResult`?**
  _Edge tagged AMBIGUOUS (relation: uses) - confidence is low._
- **What is the exact relationship between `llm_engine.py` and `bool`?**
  _Edge tagged AMBIGUOUS (relation: uses) - confidence is low._
- **What is the exact relationship between `llm_engine.py` and `float`?**
  _Edge tagged AMBIGUOUS (relation: uses) - confidence is low._
- **What is the exact relationship between `llm_engine.py` and `int`?**
  _Edge tagged AMBIGUOUS (relation: uses) - confidence is low._
- **What is the exact relationship between `llm_engine.py` and `str`?**
  _Edge tagged AMBIGUOUS (relation: uses) - confidence is low._
- **What is the exact relationship between `llm_engine.py` and `Segment`?**
  _Edge tagged AMBIGUOUS (relation: uses) - confidence is low._