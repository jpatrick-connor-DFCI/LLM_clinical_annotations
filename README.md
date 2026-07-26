# LLM clinical annotations

This repository keeps the two provider implementations in separate wrapper
directories:

- [`dfci_gpt/`](dfci_gpt/) — DFCI Azure OpenAI / GPT pipelines.
- [`vertex_ai/`](vertex_ai/) — Google Vertex AI / Gemini pipelines.

Each wrapper is self-contained and has its own `shared/` modules,
provider-specific requirements, and task directories:

```text
dfci_gpt/
  binary_NEPC/
  cancer_stage/
  gleason_score/
  longitudinal_NEPC/
  shared/
  requirements.txt

vertex_ai/
  binary_NEPC/
  cancer_stage/
  gleason_score/
  longitudinal_NEPC/
  shared/
  requirements.txt
```

Run commands from the selected wrapper directory so imports and relative paths
resolve against the correct provider implementation:

```bash
# DFCI GPT
cd dfci_gpt
python binary_NEPC/run_NEPC_classifier.py --help

# Vertex AI
cd vertex_ai
python binary_NEPC/run_NEPC_classifier.py --help
```

See the documentation within each wrapper for authentication and configuration.
