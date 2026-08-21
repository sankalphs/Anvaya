---
title: Anvaya — Grounded Voice Intelligence
emoji: 🪷
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 6.25.0
app_file: app.py
short_description: Multilingual grounded voice and text assistant
python_version: "3.12"
startup_duration_timeout: 1h
---

# Canonical deployment target

Hugging Face Space: https://sathvik0101-avyaya-voice-intelligence.hf.space/

# Anvaya

Anvaya keeps the original HTML/CSS/JS interface and runs the existing FastAPI
voice/text workflow inside a Gradio Space. The pinned BGE-M3 model downloads
on first startup; the serving FAISS index and passages are included.

Set `SARVAM_API_KEY` as a Space secret for voice input. Answer generation runs locally with
`ggml-org/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_0.gguf` inside the ZeroGPU worker.
