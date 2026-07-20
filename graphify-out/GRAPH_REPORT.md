# Graph Report - .  (2026-07-19)

## Corpus Check
- 4 files · ~14,839 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 209 nodes · 447 edges · 10 communities (7 shown, 3 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 23 edges (avg confidence: 0.73)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_CLI & Flask Entry Points|CLI & Flask Entry Points]]
- [[_COMMUNITY_Subtitle Pipeline Core|Subtitle Pipeline Core]]
- [[_COMMUNITY_LLM Backend Engine|LLM Backend Engine]]
- [[_COMMUNITY_Model Manager|Model Manager]]
- [[_COMMUNITY_Subtitle Formatter|Subtitle Formatter]]
- [[_COMMUNITY_Blackwell Setup Script|Blackwell Setup Script]]
- [[_COMMUNITY_Subtitle Formatting Guide|Subtitle Formatting Guide]]
- [[_COMMUNITY_Claude Settings|Claude Settings]]
- [[_COMMUNITY_CLI Shell Entry|CLI Shell Entry]]
- [[_COMMUNITY_Python Dependencies|Python Dependencies]]

## God Nodes (most connected - your core abstractions)
1. `run_pipeline()` - 25 edges
2. `str` - 17 edges
3. `LlamaCppBackend` - 13 edges
4. `main()` - 13 edges
5. `Path` - 12 edges
6. `segments_to_srt()` - 11 edges
7. `LLMBackend` - 10 edges
8. `OllamaBackend` - 10 edges
9. `download_youtube_audio()` - 10 edges
10. `str` - 9 edges

## Surprising Connections (you probably didn't know these)
- `SubtitleAI README Documentation` --references--> `run_pipeline()`  [EXTRACTED]
  /home/jaime/solo_ia/projects/subtitulos_srt/README.md → core/pipeline.py
- `youtube_qualities()` --calls--> `get_youtube_formats()`  [EXTRACTED]
  app_flask.py → core/pipeline.py
- `_run_job()` --calls--> `run_pipeline()`  [EXTRACTED]
  app_flask.py → core/pipeline.py
- `main()` --calls--> `Path`  [INFERRED]
  subtitles_cli.py → core/logging_setup.py
- `main()` --calls--> `is_youtube_url()`  [EXTRACTED]
  subtitles_cli.py → core/pipeline.py

## Import Cycles
- None detected.

## Communities (10 total, 3 thin omitted)

### Community 0 - "CLI & Flask Entry Points"
Cohesion: 0.08
Nodes (39): _run_job (Flask worker), _build_lang_suffix(), download(), jobs state dict, _lang_tag(), str, _run_job(), submit() (+31 more)

### Community 1 - "Subtitle Pipeline Core"
Cohesion: 0.10
Nodes (41): LLM backend priority fallback chain, core/llm_engine.py ================== Motor LLM unificado con soporte para llama, Motor LLM unificado. Selecciona automáticamente el mejor backend     disponible, analyze_audio_quality(), assign_speakers(), AudioQualityReport, correct_with_llm(), denoise_audio() (+33 more)

### Community 2 - "LLM Backend Engine"
Cohesion: 0.09
Nodes (28): ABC, detect_llama_cpp_config(), LlamaCppBackend, LLMBackend, NullBackend, OllamaBackend, bool, float (+20 more)

### Community 3 - "Model Manager"
Cohesion: 0.13
Nodes (12): VRAM-aware GGUF model selection, ModelInfo, bool, str, core/model_manager.py ===================== Gestor de modelos GGUF para llama.cp, Gestiona el ciclo de vida de los modelos GGUF locales., Lista todos los modelos del catálogo, indicando cuáles están         descargados, Retorna la ruta al mejor modelo descargado y compatible.         None si ninguno (+4 more)

### Community 4 - "Subtitle Formatter"
Cohesion: 0.22
Nodes (17): apply_punctuation_rules(), _find_split_point(), format_segments(), _min_dur_for_text(), float, int, str, core/subtitle_formatter.py ========================== Post-procesador técnico de (+9 more)

### Community 5 - "Blackwell Setup Script"
Cohesion: 0.45
Nodes (16): check_python(), download_recommended_model(), err(), info(), install_faster_whisper(), install_llama_cpp(), install_pipeline_deps(), install_pytorch() (+8 more)

### Community 6 - "Subtitle Formatting Guide"
Cohesion: 0.22
Nodes (14): CPL Rule: Max 42 Characters Per Line, CPS Rule: Max 17-20 Characters Per Second, Guia Maestra Subtitulos Professional Standards Guide, Duration Rule: 1s Min, 6-7s Max Per Subtitle, Gap Rule: Min 100ms Between Subtitles, Punctuation Rules: No Period After ?/!, Ellipsis, Dialog Dash, Pyramid Structure: Bottom Line Longer Than Top, Segmentation Rules: Syntactic Line Breaks (+6 more)

## Knowledge Gaps
- **11 isolated node(s):** `allow`, `bool`, `str`, `bool`, `float` (+6 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `run_pipeline()` connect `Subtitle Pipeline Core` to `CLI & Flask Entry Points`, `Subtitle Formatter`?**
  _High betweenness centrality (0.207) - this node is a cross-community bridge._
- **Why does `format_segments()` connect `Subtitle Formatter` to `Subtitle Pipeline Core`?**
  _High betweenness centrality (0.123) - this node is a cross-community bridge._
- **Why does `Path` connect `CLI & Flask Entry Points` to `Subtitle Pipeline Core`, `LLM Backend Engine`, `Model Manager`?**
  _High betweenness centrality (0.076) - this node is a cross-community bridge._
- **Are the 8 inferred relationships involving `Path` (e.g. with `._load()` and `.is_available()`) actually correct?**
  _`Path` has 8 INFERRED edges - model-reasoned connections that need verification._
- **What connects `allow`, `core/llm_engine.py ================== Motor LLM unificado con soporte para llama`, `Detecta el hardware disponible y devuelve la configuración     óptima para llama` to the rest of the system?**
  _70 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `CLI & Flask Entry Points` be split into smaller, more focused modules?**
  _Cohesion score 0.08078231292517007 - nodes in this community are weakly interconnected._
- **Should `Subtitle Pipeline Core` be split into smaller, more focused modules?**
  _Cohesion score 0.09745293466223699 - nodes in this community are weakly interconnected._