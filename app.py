"""Hugging Face Gradio Space entry point for the unchanged Anvaya UI."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

import spaces
from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
# The Hugging Face Space page embeds the Gradio app in an iframe. Keep the
# FastAPI security default for local deployments, while allowing this wrapper
# to be rendered by the Space host.
os.environ.setdefault("HH_RAG_ALLOW_EMBED", "1")

MODEL_REPOSITORY = "BAAI/bge-m3"
MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
QWEN_GGUF_REPOSITORY = "ggml-org/Qwen3.5-0.8B-GGUF"
QWEN_GGUF_FILENAME = "Qwen3.5-0.8B-Q4_0.gguf"
ARTIFACT_ROOT = ROOT / "space_artifacts"
MODEL_ROOT = ROOT / "cache" / "models"
SPACE_CONFIG = ARTIFACT_ROOT / "space_retriever_config.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepare_model() -> Path:
    """Download the pinned model once and add the local integrity manifest."""
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_ROOT / "bge-m3"
    snapshot_download(
        repo_id=MODEL_REPOSITORY,
        revision=MODEL_REVISION,
        local_dir=model_path,
        allow_patterns=[
            "*.json",
            "*.py",
            "*.safetensors",
            "*.bin",
            "*.pt",
            "*.model",
            "*.txt",
            "*.tiktoken",
        ],
    )
    python_hashes = {
        str(path.relative_to(model_path)).replace("\\", "/"): _sha256(path)
        for path in model_path.rglob("*.py")
        if path.is_file()
    }
    manifest = {
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "additional_code": None,
        "python_file_sha256": python_hashes,
        "owned_by": "hh-goa-retrieval-ablation-space",
    }
    (model_path / ".hh_goa_model.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return model_path


def _prepare_config(model_path: Path) -> Path:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    source_path = ROOT / "results" / "final_retriever_config.json"
    config = json.loads(source_path.read_text(encoding="utf-8"))
    config["model_cache_path"] = str(model_path.resolve())
    config["index_artifact"] = str((ARTIFACT_ROOT / "faiss_hnsw.faiss").resolve())
    config["chunk_artifact"] = str((ARTIFACT_ROOT / "serving_chunks.jsonl").resolve())
    config["model_revision"] = MODEL_REVISION
    SPACE_CONFIG.write_text(json.dumps(config, indent=2), encoding="utf-8")
    os.environ["HH_RAG_RETRIEVER_CONFIG"] = str(SPACE_CONFIG.resolve())
    os.environ.setdefault("HH_RAG_DEVICE", "cpu")
    return SPACE_CONFIG


def _prepare_qwen_model() -> Path:
    """Download the exact Qwen3.5 0.8B Q4_0 GGUF once per Space boot."""
    qwen_root = MODEL_ROOT / "qwen3.5-0.8b-q4_0"
    qwen_root.mkdir(parents=True, exist_ok=True)
    model_path = Path(
        hf_hub_download(
            repo_id=QWEN_GGUF_REPOSITORY,
            filename=QWEN_GGUF_FILENAME,
            local_dir=qwen_root,
        )
    )
    os.environ["QWEN_GGUF_PATH"] = str(model_path.resolve())
    return model_path


def _body_markup() -> str:
    source = (ROOT / "src" / "hh_goa_rag" / "static" / "index.html").read_text(
        encoding="utf-8"
    )
    body = re.search(r"<body[^>]*>(?P<body>.*)</body>", source, flags=re.DOTALL | re.IGNORECASE)
    if body is None:
        raise RuntimeError("The existing Anvaya HTML shell has no body")
    return re.sub(
        r"\s*<script[^>]+src=[\"']/static/app\.js[\"'][^>]*></script>",
        "",
        body.group("body"),
        flags=re.IGNORECASE,
    )


@spaces.GPU(duration=1)
def _space_bootstrap() -> None:
    """Reserve the required ZeroGPU hook without moving pipeline work to GPU."""


def _space_harness_factory(_settings):
    """Prepare Space-local artifacts and construct the CPU serving harness."""
    from hh_goa_rag.harness import VoiceRAGHarness
    from hh_goa_rag.web import AppSettings, validate_environment

    model_path = _prepare_model()
    config_path = _prepare_config(model_path)
    _prepare_qwen_model()
    runtime_settings = AppSettings.from_env()
    runtime_settings = AppSettings(
        retriever_config=config_path,
        env_file=runtime_settings.env_file,
        device="cpu",
        api_token=runtime_settings.api_token,
    )
    validate_environment(runtime_settings)
    return VoiceRAGHarness.from_frozen_artifacts(
        retriever_config_path=config_path,
        env_path=runtime_settings.env_file,
        device="cpu",
        include_stt=True,
    )


def _build_space_app():
    import gradio as gr
    from fastapi.responses import RedirectResponse

    from hh_goa_rag.web import create_app

    styles = (ROOT / "src" / "hh_goa_rag" / "static" / "styles.css").read_text(
        encoding="utf-8"
    )
    script = (ROOT / "src" / "hh_goa_rag" / "static" / "app.js").read_text(
        encoding="utf-8"
    )
    gradio_reset = """
    :root { color-scheme: light !important; }
      html, body, body.dark { min-height: 0 !important; height: auto !important; margin: 0 !important; padding: 0 !important; background-color: #f4f1e8 !important; color: #17211b !important; color-scheme: light !important; }
      gradio-app { background: transparent !important; color: #17211b !important; }
      body { overflow-x: hidden !important; overflow-y: auto !important; background-image: radial-gradient(circle 224px at -160px -192px, rgba(199, 221, 197, .75) 0 100%, transparent 100%), radial-gradient(circle 272px at calc(100% + 288px) 304px, rgba(231, 211, 177, .65) 0 100%, transparent 100%) !important; background-repeat: no-repeat !important; background-attachment: fixed !important; }
      .gradio-container, .gradio-container > .main, .gradio-container .contain, .gradio-container .block { min-height: 0 !important; height: auto !important; max-width: none !important; padding: 0 !important; margin: 0 !important; background: transparent !important; color: #17211b !important; }
      .html-container { min-height: 0 !important; height: auto !important; padding: 0 !important; }
      .ambient, html > .ambient { position: fixed !important; z-index: 0 !important; }
      .page-shell { position: relative !important; z-index: 1 !important; color: #17211b !important; }
      .page-shell .brand, .page-shell .brand-mark, .page-shell .brand > span:not(.brand-type) { color: #17211b !important; }
      .page-shell .hero-trust-row span:not(.trust-icon) { color: #667068 !important; }
      .page-shell .hero h1, .page-shell .query-heading h2 { color: #17211b !important; }
      .page-shell .upload-control strong { color: #17211b !important; }
      .page-shell .transparency-card h2 { color: #edf4ee !important; }
      .page-shell .trust-link, .page-shell .system-chip, .page-shell .system-chip #health-text { color: #667068 !important; }
      .language-control select, .language-control select option, .language-dropdown-button, .language-dropdown-option { color: #17211b !important; background: #fffffb !important; color-scheme: light !important; }
    #text-query { color: #17211b !important; background: #fffffb !important; color-scheme: light !important; }
    footer { display: none !important; }
    """
    space_js = (
        "() => {\n"
        "  const forceLightTheme = () => {\n"
        "    document.documentElement.classList.remove('dark');\n"
        "    document.body.classList.remove('dark');\n"
        "  };\n"
        "  forceLightTheme();\n"
        "  new MutationObserver(forceLightTheme).observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });\n"
        "  new MutationObserver(forceLightTheme).observe(document.body, { attributes: true, attributeFilter: ['class'] });\n"
        "  let appStarted = false;\n"
        "  const startAnvaya = () => {\n"
        "    if (appStarted) return;\n"
        "    if (!document.querySelector('#record-button')) { window.setTimeout(startAnvaya, 50); return; }\n"
        "    appStarted = true;\n"
        + "    " + script.replace("\n", "\n    ")
        + "\n  };\n"
        "  startAnvaya();\n"
        "\n}"
    )
    with gr.Blocks(
        title="Anvaya — Grounded Voice Intelligence",
        theme=gr.themes.Base(),
    ) as demo:
        gr.HTML(_body_markup())
        # ZeroGPU requires at least one GPU-decorated Gradio event. The frozen
        # pipeline remains CPU-bound; this no-op keeps the Space eligible
        # without changing the existing UI or request workflow.
        zero_gpu_trigger = gr.Button(visible=False, elem_id="zerogpu-bootstrap")
        zero_gpu_trigger.click(
            _space_bootstrap,
            inputs=[],
            outputs=[],
            api_name="zerogpu_bootstrap",
        )
        demo.load(js=space_js)

    api_app = create_app(
        harness_factory=_space_harness_factory,
        include_ui=False,
        defer_startup=True,
    )
    @api_app.get("/", include_in_schema=False)
    async def space_home() -> RedirectResponse:
        return RedirectResponse("/ui/", status_code=307)

    return gr.mount_gradio_app(
        api_app,
        demo,
        path="/ui",
        ssr_mode=False,
        css=styles + gradio_reset,
    )


app = _build_space_app()


if __name__ == "__main__":
    import uvicorn
    from spaces import zero

    zero.startup()
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
