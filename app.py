"""
app.py
------
A small Streamlit front-end over the existing pipeline (src/pipeline.py).
This is a thin UI layer only -- all the actual retrieval/scoring logic
still lives in src/, exactly as it does for the CLI (main.py --score). The
app just gives a paste-a-conversation-and-see-results experience instead of
a terminal flag.

Run with:
    streamlit run app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make sure src/ is importable regardless of the working directory Streamlit
# was launched from (mirrors the same pattern used in main.py / the notebook).
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Ahoum Facet Scoring Pipeline", page_icon="🧭", layout="wide")


def check_ollama_status() -> tuple[bool, str]:
    """
    Lightweight Ollama reachability check, used to show a status badge and
    to fail fast with a clear message instead of letting the pipeline hang
    or throw a confusing low-level connection error mid-run.
    Returns (is_ready, message).
    """
    try:
        import ollama
        client = ollama.Client(host="http://localhost:11434")
        models_response = client.list()
        models = models_response.get("models", []) if isinstance(models_response, dict) else models_response.models
        model_names = []
        for m in models:
            name = m.get("model") if isinstance(m, dict) else getattr(m, "model", None)
            if name:
                model_names.append(name)
        if any("llama3.1" in name for name in model_names):
            return True, "Ollama is running and llama3.1 is available."
        return False, "Ollama is running, but llama3.1 isn't pulled yet. Run: ollama pull llama3.1"
    except Exception as e:
        return False, (
            "Can't reach Ollama at http://localhost:11434 -- is it running? "
            f"(details: {e})"
        )


def check_index_status() -> tuple[bool, str]:
    """Confirm the FAISS index / enriched facets exist before letting the
    user try to score anything -- otherwise retrieval fails with a much
    less friendly FileNotFoundError deep inside embeddings.py."""
    faiss_path = PROJECT_ROOT / "outputs" / "faiss_index.bin"
    facets_path = PROJECT_ROOT / "outputs" / "observable_facets.json"
    if faiss_path.exists() and facets_path.exists():
        return True, "Facet index found."
    return False, (
        "No FAISS index found in outputs/. Run `python main.py --audit` "
        "then `python main.py --embed` first."
    )


def check_gpu_vram_status() -> tuple[str, str]:
    """
    Checks free GPU VRAM via torch. This exists because of a real crash
    found during testing (docs/DEBUGGING.md #3): Ollama's llama-server
    inference worker crashed mid-batch when the Jupyter kernel, browser,
    and Streamlit were all competing for the RTX 4050's 6GB of VRAM at
    once. Surfacing this *before* a run starts is a lot more useful than a
    wall of parse_error rows after the fact.

    Returns (level, message) where level is one of
    "green" / "yellow" / "red" / "unknown".

    Note: this uses torch.cuda.mem_get_info(), not
    torch.cuda.get_device_properties(0).total_memory. total_memory is the
    card's fixed capacity (always ~6GB on this laptop, for example) and
    doesn't change no matter what else is using the GPU -- it would never
    actually detect the contention this badge is meant to warn about.
    mem_get_info() returns the CURRENTLY FREE bytes on the device, which is
    what the green/yellow/red thresholds below are actually about.
    """
    try:
        import torch
    except ImportError:
        return "unknown", "GPU status unknown -- torch isn't installed."

    try:
        # A CPU-only torch build (pip installs this by default on Windows
        # unless you point at the CUDA wheel index) always reports
        # is_available()=False -- even when a real NVIDIA GPU is present
        # and Ollama itself is actively using it via its own CUDA runtime.
        # Reporting "red / no GPU" in that case would be a false alarm, so
        # this is treated as "can't check" rather than "no GPU" -- run
        # `nvidia-smi` yourself to check VRAM directly if you see this.
        if getattr(torch.version, "cuda", None) is None:
            return "unknown", (
                "GPU status unknown -- this torch install has no CUDA support "
                "(pip installed the CPU-only build), so it can't see the GPU "
                "even if one is present and in use by Ollama. Run `nvidia-smi` "
                "to check VRAM directly."
            )
        if not torch.cuda.is_available():
            return "red", "No CUDA GPU detected."
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        free_gb = free_bytes / (1024 ** 3)
        total_gb = total_bytes / (1024 ** 3)
        if free_gb > 3:
            return "green", f"{free_gb:.1f} GB free / {total_gb:.1f} GB total."
        if free_gb >= 1:
            return "yellow", (
                f"Only {free_gb:.1f} GB free / {total_gb:.1f} GB total -- close other "
                "GPU workloads (Jupyter kernels, extra browser tabs) for the most "
                "reliable results."
            )
        return "red", (
            f"Only {free_gb:.1f} GB free / {total_gb:.1f} GB total -- Ollama may crash "
            "mid-batch under this little headroom (see docs/DEBUGGING.md #3)."
        )
    except Exception as e:
        return "unknown", f"GPU status unknown ({e})."


# ---------------------------------------------------------------------------
# Header + status badges
# ---------------------------------------------------------------------------
st.title("🧭 Ahoum Facet Scoring Pipeline")
st.caption(
    "Paste a short conversation below. The system retrieves the most relevant "
    "personality facets (out of 399) via FAISS, then scores them in small "
    "batches with a local llama3.1 model -- abstaining explicitly wherever "
    "the conversation doesn't actually support a judgment."
)

index_ok, index_msg = check_index_status()
ollama_ok, ollama_msg = check_ollama_status()
gpu_level, gpu_msg = check_gpu_vram_status()

status_col1, status_col2, status_col3 = st.columns(3)
with status_col1:
    if index_ok:
        st.success(f"✅ Facet index: {index_msg}")
    else:
        st.error(f"❌ Facet index: {index_msg}")
with status_col2:
    if ollama_ok:
        st.success(f"✅ Ollama: {ollama_msg}")
    else:
        st.warning(f"⚠️ Ollama: {ollama_msg}")
with status_col3:
    if gpu_level == "green":
        st.success(f"✅ GPU VRAM: {gpu_msg}")
    elif gpu_level == "yellow":
        st.warning(f"⚠️ GPU VRAM: {gpu_msg}")
    elif gpu_level == "red":
        st.error(f"❌ GPU VRAM: {gpu_msg}")
    else:
        st.info(f"❔ GPU VRAM: {gpu_msg}")

st.divider()

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None

conversation = st.text_area(
    "Conversation",
    height=180,
    placeholder=(
        "e.g. \"I quit my stable job last week to backpack solo through South "
        "America with no itinerary and barely any savings left...\""
    ),
)

run_clicked = st.button("Run Pipeline", type="primary", disabled=not index_ok)

if not index_ok:
    st.info("Build the facet index first (see the error above) before running the pipeline.")

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if run_clicked:
    if not conversation or not conversation.strip():
        st.warning("Please paste a conversation first.")
    else:
        with st.spinner("Retrieving relevant facets and scoring with llama3.1 (this can take a minute)..."):
            try:
                from src.pipeline import run_pipeline
                result = run_pipeline(conversation, save_output=True)
                st.session_state.pipeline_result = result
            except FileNotFoundError as e:
                st.session_state.pipeline_result = None
                st.error(
                    "Missing a required file. Make sure you've run "
                    "`python main.py --audit` and `python main.py --embed` first.\n\n"
                    f"Details: {e}"
                )
            except Exception as e:
                st.session_state.pipeline_result = None
                # Covers "Ollama isn't running" (connection refused), a model
                # that isn't pulled, or any other unexpected failure -- we
                # never want the app itself to crash on a bad LLM call.
                st.error(
                    "Something went wrong while running the pipeline. This "
                    "usually means Ollama isn't running or llama3.1 isn't "
                    "pulled yet.\n\n"
                    f"Details: {e}"
                )

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
result = st.session_state.pipeline_result

if result is not None:
    if result.get("error"):
        st.error(f"Pipeline reported an error: {result['error']}")
    else:
        st.subheader("Summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Facets retrieved", result.get("total_facets_retrieved", 0))
        m2.metric("Scored", result.get("scored", 0))
        m3.metric("Abstained", result.get("abstained", 0))
        m4.metric("Parse errors", result.get("parse_errors", 0))

        results_list = result.get("results", [])

        if not results_list:
            st.info("No facets were retrieved for this conversation -- try a longer or more descriptive one.")
        else:
            df = pd.DataFrame(results_list)
            # Keep a stable, readable column order regardless of dict key order.
            column_order = [c for c in ["facet", "score", "status", "confidence", "evidence"] if c in df.columns]
            df = df[column_order]

            st.subheader("Facet scores")
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.subheader("Outcome distribution")
            status_counts = df["status"].value_counts()
            st.bar_chart(status_counts)

            with st.expander("Raw JSON output"):
                st.json(result)
else:
    st.caption("Results will appear here after you run the pipeline.")
