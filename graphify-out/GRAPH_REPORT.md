# Graph Report - .  (2026-07-05)

## Corpus Check
- Corpus is ~14,709 words - fits in a single context window. You may not need a graph.

## Summary
- 223 nodes · 457 edges · 10 communities (7 shown, 3 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 24 edges (avg confidence: 0.74)
- Token cost: 84,965 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_LLM Backend Strategy|LLM Backend Strategy]]
- [[_COMMUNITY_Transcription Pipeline|Transcription Pipeline]]
- [[_COMMUNITY_Flask Web App & YouTube|Flask Web App & YouTube]]
- [[_COMMUNITY_GGUF Model Manager|GGUF Model Manager]]
- [[_COMMUNITY_Subtitle Formatting|Subtitle Formatting]]
- [[_COMMUNITY_Environment Setup|Environment Setup]]
- [[_COMMUNITY_Subtitle Standards Guide|Subtitle Standards Guide]]
- [[_COMMUNITY_Claude Permissions Config|Claude Permissions Config]]
- [[_COMMUNITY_CLI Run Script|CLI Run Script]]
- [[_COMMUNITY_Python Requirements|Python Requirements]]

## God Nodes (most connected - your core abstractions)
1. `LLMEngine` - 30 edges
2. `run_pipeline()` - 28 edges
3. `str` - 18 edges
4. `str` - 17 edges
5. `LlamaCppBackend` - 14 edges
6. `ModelManager` - 13 edges
7. `Path` - 12 edges
8. `main()` - 12 edges
9. `OllamaBackend` - 11 edges
10. `segments_to_srt()` - 11 edges

## Surprising Connections (you probably didn't know these)
- `SubtitleAI README Documentation` --references--> `LLMEngine`  [EXTRACTED]
  /home/jaime/solo_ia/projects/subtitulos_srt/README.md → core/llm_engine.py
- `SubtitleAI README Documentation` --references--> `run_pipeline()`  [EXTRACTED]
  /home/jaime/solo_ia/projects/subtitulos_srt/README.md → core/pipeline.py
- `_build_lang_suffix (CLI)` --semantically_similar_to--> `_build_lang_suffix (Flask)`  [INFERRED] [semantically similar]
  subtitles_cli.py → app_flask.py
- `youtube_qualities()` --calls--> `get_youtube_formats()`  [EXTRACTED]
  app_flask.py → core/pipeline.py
- `_run_job()` --calls--> `run_pipeline()`  [EXTRACTED]
  app_flask.py → core/pipeline.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Guia Maestra Technical Subtitle Standards (CPL, CPS, Gap, Duration, Pyramid)** — guia_maestra_subtitulos_cpl_rule, guia_maestra_subtitulos_cps_rule, guia_maestra_subtitulos_gap_rule, guia_maestra_subtitulos_duration_rule, guia_maestra_subtitulos_pyramid_structure [EXTRACTED 1.00]
- **LLM backend Strategy pattern (llama.cpp / Ollama / Null)** — core_llm_engine_llmbackend, core_llm_engine_llamacppbackend, core_llm_engine_ollamabackend, core_llm_engine_nullbackend, core_llm_engine_llmengine [EXTRACTED 1.00]
- **Subtitle generation pipeline flow** — core_pipeline_preprocess_audio, core_pipeline__transcribe_with_beam, core_pipeline_diarize, core_pipeline_correct_with_llm, core_pipeline_translate_with_llm [INFERRED 0.85]
- **Segment to SRT serialization** — core_pipeline_segment, core_pipeline_segments_to_srt, core_pipeline_seconds_to_srt [INFERRED 0.85]

## Communities (10 total, 3 thin omitted)

### Community 0 - "LLM Backend Strategy"
Cohesion: 0.07
Nodes (28): ABC, detect_llama_cpp_config(), LlamaCppBackend, LLMBackend, NullBackend, OllamaBackend, bool, float (+20 more)

### Community 1 - "Transcription Pipeline"
Cohesion: 0.08
Nodes (46): LLM backend priority fallback chain, LLMEngine, Motor LLM unificado. Selecciona automáticamente el mejor backend     disponible, Descarga el modelo para liberar VRAM antes de cargar Whisper., _denoise_with_strength, _detect_language_fast, _transcribe_with_beam, analyze_audio_quality() (+38 more)

### Community 2 - "Flask Web App & YouTube"
Cohesion: 0.08
Nodes (34): _build_lang_suffix (Flask), _run_job (Flask worker), _update_job, _build_lang_suffix(), download(), jobs state dict, _lang_tag(), str (+26 more)

### Community 3 - "GGUF Model Manager"
Cohesion: 0.13
Nodes (13): VRAM-aware GGUF model selection, ModelInfo, ModelManager, bool, str, core/model_manager.py ===================== Gestor de modelos GGUF para llama.cp, Gestiona el ciclo de vida de los modelos GGUF locales., Lista todos los modelos del catálogo, indicando cuáles están         descargados (+5 more)

### Community 4 - "Subtitle Formatting"
Cohesion: 0.22
Nodes (17): apply_punctuation_rules(), _find_split_point(), format_segments(), _min_dur_for_text(), float, int, str, core/subtitle_formatter.py ========================== Post-procesador técnico de (+9 more)

### Community 5 - "Environment Setup"
Cohesion: 0.45
Nodes (16): check_python(), download_recommended_model(), err(), info(), install_faster_whisper(), install_llama_cpp(), install_pipeline_deps(), install_pytorch() (+8 more)

### Community 6 - "Subtitle Standards Guide"
Cohesion: 0.22
Nodes (14): CPL Rule: Max 42 Characters Per Line, CPS Rule: Max 17-20 Characters Per Second, Guia Maestra Subtitulos Professional Standards Guide, Duration Rule: 1s Min, 6-7s Max Per Subtitle, Gap Rule: Min 100ms Between Subtitles, Punctuation Rules: No Period After ?/!, Ellipsis, Dialog Dash, Pyramid Structure: Bottom Line Longer Than Top, Segmentation Rules: Syntactic Line Breaks (+6 more)

## Knowledge Gaps
- **10 isolated node(s):** `allow`, `bool`, `str`, `bool`, `float` (+5 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `LLMEngine` connect `Transcription Pipeline` to `LLM Backend Strategy`, `Flask Web App & YouTube`, `GGUF Model Manager`?**
  _High betweenness centrality (0.289) - this node is a cross-community bridge._
- **Why does `run_pipeline()` connect `Transcription Pipeline` to `Flask Web App & YouTube`, `Subtitle Formatting`?**
  _High betweenness centrality (0.226) - this node is a cross-community bridge._
- **Why does `format_segments()` connect `Subtitle Formatting` to `Transcription Pipeline`?**
  _High betweenness centrality (0.117) - this node is a cross-community bridge._
- **Are the 10 inferred relationships involving `LLMEngine` (e.g. with `VRAM-aware GGUF model selection` and `LLM backend priority fallback chain`) actually correct?**
  _`LLMEngine` has 10 INFERRED edges - model-reasoned connections that need verification._
- **What connects `allow`, `core/llm_engine.py ================== Motor LLM unificado con soporte para llama`, `Detecta el hardware disponible y devuelve la configuración     óptima para llama` to the rest of the system?**
  _67 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `LLM Backend Strategy` be split into smaller, more focused modules?**
  _Cohesion score 0.06531986531986532 - nodes in this community are weakly interconnected._
- **Should `Transcription Pipeline` be split into smaller, more focused modules?**
  _Cohesion score 0.08408163265306122 - nodes in this community are weakly interconnected._